import rclpy 
from rclpy.node import Node
import pyzed.sl as sl
import torch 
import sys
from torchvision import transforms
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
import numpy as np 

#load the lane segmentation model
sys.path.append('.../LFD_RoadSeg') # fix later 
from models.LFD_RoadSeg import LFD_RoadSeg


class LaneSegmentationNode(Node):
    def __init__(self, name='lane_segmentation_node'):
        super().__init__(name)
        self.lane_segmentation = None

        # make object
        self.zed = sl.Camera()

        # config parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 25

        err = self.zed.open(init_params)
        if (err != sl.ERROR_CODE.SUCCESS) :
            exit(-1)

        self.image_zed = sl.Mat()
        timer_period = 1.0 / 25.0  # 25 fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('ZED Node Started successfully')

        self.get_logger().info('Loading custom LFD_RoadSeg Model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LFD_RoadSeg(num_classes = 2)
        weights_path = '.../LFD_RoadSeg/weights/best.pt' # fix later
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.get_logger().info('Model loaded successfully')

        self.publisher_ = self.create_publisher(Image, 'lane_mask', 10)
        self.bridge = CvBridge()


    def timer_callback(self):

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            image_data = self.image_zed.get_data() #numpy array 
            
            image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
            target_training_size = (1248, 384) # fix later (should be same as training size)
            image_resized = cv2.resize(image_rgb, target_training_size)

            input_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            visual_mask = (predicted_mask * 255).astype(np.uint8)
            ros_image = self.bridge.cv2_to_imgmsg(visual_mask, encoding="mono8")

            self.publisher_.publish(ros_image)
            self.get_logger().info('Image grabbed and processed and released succesfully')

        
def main(args=None):
    rclpy.init(args=args)
    node = LaneSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.zed.close()
        node.destroy_node()
        rclpy.shutdown()
