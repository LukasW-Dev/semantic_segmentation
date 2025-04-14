import rclpy
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
        self.declare_parameter('config_file', '/home/lukas/ros2_ws/src/semantic_segmentation/model_config/mask2former/mask2former_swin-l-in22k-384x384-pre_2xb20-80k_wildscenes_standard-512x512.py')
        self.declare_parameter('checkpoint_file', '/home/lukas/ros2_ws/src/semantic_segmentation/pretrained_models/mask2former_swin_wildscenes.pth')
        self.declare_parameter('input_topic', '/hazard_front/zed_node_front/left/image_rect_color')
        self.declare_parameter('output_topic', '/segmentation/image')

        # Get parameters
        self.config_file = self.get_parameter('config_file').get_parameter_value().string_value
        self.checkpoint_file = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value


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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = init_model(self.config_file, self.checkpoint_file, device=device)
        self.bridge = CvBridge()
        
        # Subscribe to image topic
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self.image_callback, 0)

        # Publisher for segmented image
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)

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

        # Sort the logits along the class dimension
        _, sorted_indices = torch.sort(seg_logits, dim=0, descending=True)

        # Get second-best prediction per pixel
        second_best = sorted_indices[1]  # shape: [H, W]

        # Find mask where target_class is predicted
        mask = (pred_seg == target_class)

        # Replace class in pred_seg
        pred_seg[mask] = second_best[mask]

        return pred_seg
    
    def image_callback(self, msg):
        self.get_logger().info("Received an image. Processing...")

        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        result = inference_model(self.model, img_rgb)

        # Get the predicted class map (pred_seg is of shape [720, 1280])
        pred_seg = result.pred_sem_seg.data[0].cpu().unsqueeze(0)
        self.get_logger().info(f"pred_seg shape: {pred_seg.shape}")

        # Get the logits for all classes (seg_logits is of shape [720, 1280] for all pixels)
        seg_logits = result.seg_logits.data.cpu()
        self.get_logger().info(f"seg_logits shape: {seg_logits.shape}")

        # Replace class 7 with second-highest prediction
        pred_seg = self.replace_class_with_second_highest(pred_seg, seg_logits, 7)

        # Convert to numpy for visualization
        segmentation_map = pred_seg.numpy()

        # Create an overlay
        segmentation_colored = np.zeros_like(frame, dtype=np.uint8)
        for label, color in self.color_map.items():
            segmentation_colored[np.where(segmentation_map == label)] = color

        # Blend segmentation mask with the original frame
        alpha = 0.8
        blended_frame = cv2.addWeighted(frame, 1 - alpha, segmentation_colored, alpha, 0)

        # Convert OpenCV image back to ROS Image
        segmented_msg = self.bridge.cv2_to_imgmsg(blended_frame, encoding='rgb8')
        segmented_msg.header = msg.header

        # Publish the segmented image
        self.image_pub.publish(segmented_msg)
        self.get_logger().info("Published segmented image.")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
