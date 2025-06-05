import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from mmseg.apis import inference_model, init_model

class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation_node')
        
        # Declare parameters with default values in case they are not set
        self.declare_parameter('config_file', None)
        self.declare_parameter('checkpoint_file', None)
        self.declare_parameter('input_topic', None)
        self.declare_parameter('output_topic', None)
        self.declare_parameter('confidence_topic', None)

        # Get parameters
        self.config_file = os.path.expanduser(self.get_parameter('config_file').get_parameter_value().string_value)
        self.checkpoint_file = os.path.expanduser(self.get_parameter('checkpoint_file').get_parameter_value().string_value)
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.confidence_topic = self.get_parameter('confidence_topic').get_parameter_value().string_value


        # Define color mappings
        self.color_map = {
            0:  [34, 139, 34],    # (bush)
            1: 	[139, 69, 19],    # (dirt)
            2:  [255, 215, 0],    # (fence)
            3:  [124, 252, 0],    # (grass)
            4:  [169, 169, 169],  # (gravel)
            5:  [160, 82, 45],    # (log)
            6: 	[101, 67, 33],    # (mud)
            7:  [255, 0, 255],    # (object)
            8:  [128, 128, 0],    # (other-terrain)
            9:  [112, 128, 144],  # (rock)
            10: [135, 206, 235],  # (sky)
            11: [178, 34, 34],    # (structure)
            12: [0, 100, 0],      # (tree-foliage)
            13: [139, 115, 85],   # (tree-trunk)
            14: [0, 191, 255],    # (water)
            15: [0, 0, 0],        # (unlabeled)
            16: [0, 0, 0],        # (unlabeled)
            17: [0, 0, 0],        # (unlabeled)
            18: [0, 0, 0],        # (unlabeled)
        }

        # Initialize the model
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = init_model(self.config_file, self.checkpoint_file, device=device_str)
        self.device = torch.device(device_str)
        self.bridge = CvBridge()
        
        # Subscribe to image topic
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, 0)

        # Publisher for segmented image
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)
        # Publisher for confidence map
        self.confidence_pub = self.create_publisher(Image, self.confidence_topic, 10)

        self.get_logger().info(f"SemanticSegmentationNode initialized. Subscribing to {self.input_topic}")

    def replace_class_with_second_highest(self, pred_seg: torch.Tensor, seg_logits: torch.Tensor, target_class: int = 7) -> torch.Tensor:
        """
        Replaces pixels in pred_seg that are equal to target_class with the second-highest predicted class
        based on the logits in seg_logits.

        Args:
            pred_seg (torch.Tensor): Tensor of shape [H, W] with predicted class indices.
            seg_logits (torch.Tensor): Tensor of shape [C, H, W] with class logits.
            target_class (int): The class to be replaced.

        Returns:
            torch.Tensor: Updated pred_seg with replaced classes.
        """
        # Sanity check shapes
        assert seg_logits.dim() == 3, "seg_logits should be [C, H, W]"
        pred_seg = pred_seg.squeeze(dim=0)
        assert pred_seg.shape == seg_logits.shape[1:], "pred_seg shape must match spatial dims of seg_logits"

        # # Sort the logits along the class dimension
        # _, sorted_indices = torch.sort(seg_logits, dim=0, descending=True)

        # # Get second-best prediction per pixel
        # second_best = sorted_indices[1]  # shape: [H, W]

        # Instead of sort, use topk(k=2) to get the indices of the top-2 classes per pixel
        # seg_logits has shape [C, H, W]
        logits_flat = seg_logits.view(seg_logits.shape[0], -1)  # [C, H*W]
        # top2_vals, top2_idx_each = torch.topk(logits_flat, k=2, dim=0)
        # → top2_idx_each is shape [2, H*W]. Reshape back to [2, H, W]:
        top2_idx_each = torch.topk(seg_logits, k=2, dim=0)[1]  # directly on [C, H, W] dims
        second_best = top2_idx_each[1]  # shape [H, W]

        # Find mask where target_class is predicted
        mask = (pred_seg == target_class)

        # Replace class in pred_seg
        pred_seg[mask] = second_best[mask]

        return pred_seg


    def image_callback(self, msg):
        # # Measure Callback Time
        # start_time = self.get_clock().now()

        # Convert ROS Image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize once (1280×720)
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # 1. RUN INFERENCE (on GPU)
        # -----------------------------
        # Wrap in no_grad() to avoid any grad‐tracking overhead
        with torch.no_grad():
            result = inference_model(self.model, img_rgb)
            # result.pred_sem_seg likely already on GPU, but to be sure:
            pred_seg = result.pred_sem_seg.data[0].unsqueeze(0).to(self.device)   # [1, H, W]
            seg_logits = result.seg_logits.data.to(self.device)                   # [C, H, W]

        # -----------------------------
        # 2. REPLACE CLASS 7 ON GPU
        # -----------------------------
        # (Assumes replace_class_with_second_highest works with GPU tensors)
        pred_seg = self.replace_class_with_second_highest(pred_seg, seg_logits, target_class=7)
        # Now pred_seg is [1, H, W], seg_logits is [C, H, W], all on self.device

        # -----------------------------
        # 3. VECTORIZE COLOR + CONFIDENCE (ALL ON GPU)
        # -----------------------------

        # 3a. Build a palette‐tensor for all labels (shape [C, 3], dtype=uint8, on GPU)
        #     We assume color_map keys from 0..18 in order
        palette_list = [self.color_map[i] for i in range(len(self.color_map))]
        palette = torch.tensor(palette_list, device=self.device, dtype=torch.uint8)  # [C, 3]

        # 3b. Get height/width
        _, H, W = seg_logits.shape

        # 3c. Build the colored segmentation (H×W×3) with one gather
        pred_flat = pred_seg.squeeze(0).long()                        # [H, W], values in 0..C-1
        colored_tensor = palette[pred_flat]                            # [H, W, 3], uint8, on GPU

        # 3d. Normalize logits to [0..100] per-channel, then gather confidence per-pixel
        mins = seg_logits.amin(dim=(1, 2), keepdim=True)               # [C, 1, 1]
        maxs = seg_logits.amax(dim=(1, 2), keepdim=True)               # [C, 1, 1]
        normed = ((seg_logits - mins) / (maxs - mins + 1e-6)) * 100.0   # [C, H, W], float32
        normed = normed.to(torch.uint8)                                # [C, H, W], uint8

        # Flatten spatial dims so we can index in one shot
        flat_idx = pred_flat.view(-1)                                  # [H*W]
        flat_normed = normed.view(seg_logits.shape[0], -1)             # [C, H*W]
        device_idx = torch.arange(H * W, device=self.device)           # [H*W]
        flat_conf = flat_normed[flat_idx, device_idx]                  # [H*W], uint8
        conf_map_tensor = flat_conf.view(H, W)                         # [H, W], uint8, on GPU

        # -----------------------------
        # 4. MOVE TO CPU ONCE
        # -----------------------------
        segmentation_colored = colored_tensor.cpu().numpy()            # [H, W, 3], uint8 BGR
        confidence_map = conf_map_tensor.cpu().numpy()                 # [H, W], uint8

        # -----------------------------
        # 5. PUBLISH
        # -----------------------------
        # Convert confidence_map to ROS Image (mono8)
        confidence_msg = self.bridge.cv2_to_imgmsg(confidence_map, encoding='mono8')
        confidence_msg.header = msg.header

        # Blend segmentation overlay with original frame (both are BGR)
        alpha = 1.0
        blended_frame = cv2.addWeighted(frame, 1 - alpha, segmentation_colored, alpha, 0)
        # Convert blended_frame (BGR) → ROS Image as RGB8 (cv_bridge will reorder channels)
        segmented_msg = self.bridge.cv2_to_imgmsg(blended_frame, encoding='rgb8')
        segmented_msg.header = msg.header

        self.image_pub.publish(segmented_msg)
        self.confidence_pub.publish(confidence_msg)

        # # Log the processing time
        # end_time = self.get_clock().now()
        # elapsed_time = (end_time - start_time).nanoseconds / 1e6
        # self.get_logger().info(f"Processed image in {elapsed_time:.2f} ms")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
