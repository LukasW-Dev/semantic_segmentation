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
        self.declare_parameter('config_file', '')
        self.declare_parameter('checkpoint_file', '')
        self.declare_parameter('input_topic', '')
        self.declare_parameter('output_topic', '')

        # Get parameters
        self.config_file = self.get_parameter('config_file').get_parameter_value().string_value
        self.checkpoint_file = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value


        # Define color mappings
        self.color_map = {
            0: [51, 153, 51],     # (bush)
            1: [153, 77, 0],      # (dirt)
            2: [60, 180, 75],     # 
            3: [0, 204, 0],       # (grass)
            4: [230, 230, 230],   # (gravel?)
            5: [128, 0, 0],       # (branch)
            6: [70, 240, 240],    # 
            7: [240, 50, 230],    # (fence)
            8: [210, 245, 60],    # 
            9: [230, 25, 75],     # 
            10: [51, 153, 255],   # (sky)
            11: [170, 110, 40],   # 
            12: [0, 77, 0],       # (tree-foliage)
            13: [128, 0, 0],      # (tree)
            14: [170, 255, 195],  # 
            15: [128, 128, 0],    # 
            16: [250, 190, 190],  # 
            17: [0, 0, 128],      # 
            18: [128, 128, 128],  # 
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
    
    def image_callback(self, msg):
        self.get_logger().info("Received an image. Processing...")
        
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        result = inference_model(self.model, img_rgb)
        segmentation_map = result.pred_sem_seg.data[0].cpu().numpy()
        
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
