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
import os 


#load the lane segmentation model
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../.."))

lfd_repo_path = os.path.join(src_path, "LFD_RoadSeg")
if lfd_repo_path not in sys.path:
    sys.path.append(lfd_repo_path)
from models._LFDRoadSeg import LFD_RoadSeg


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

        self.image_zed_left = sl.Mat()
        self.image_zed_right = sl.Mat()

        timer_period = 1.0 / 25.0  # 25 fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('ZED Node Started successfully')

        cfg = {
            'models': {'backbone': 'resnet18'},
            'training': {'scale_factor': 2} 
        }
        
        self.get_logger().info('Loading custom LFD_RoadSeg Model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LFD_RoadSeg(cfg)

        weights_path = 'model_epoch_150.pth'

        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3598, 0.3653, 0.3662], 
                std=[0.2573, 0.2663, 0.2756]
            )
        ])

        self.get_logger().info('Model loaded successfully')

        self.pub_left = self.create_publisher(Image, 'lane_mask/left', 10)
        self.pub_right = self.create_publisher(Image, 'lane_mask/right', 10)
        self.bridge = CvBridge()

    def letterbox_image(self, image, target_size=(1248, 384)):
        target_w, target_h = target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        padded_img[top:top+new_h, left:left+new_w] = resized_img

        return padded_img, (scale, top, left, new_w, new_h)

    def unletterbox_mask(self, mask, padding_info, original_size):
        scale, top, left, new_w, new_h = padding_info
        orig_w, orig_h = original_size
        cropped_mask = mask[top:top+new_h, left:left+new_w]
        restored_mask = cv2.resize(cropped_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return restored_mask
    
    def timer_callback(self):

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            
            self.zed.retrieve_image(self.image_zed_left, sl.VIEW.LEFT)

            self.zed.retrieve_image(self.image_zed_right, sl.VIEW.RIGHT)

            image_data_left = self.image_zed_left.get_data() #numpy array 
            image_data_right = self.image_zed_right.get_data() #numpy array

            
            image_rgb_left = cv2.cvtColor(image_data_left, cv2.COLOR_BGRA2RGB)
            image_rgb_right = cv2.cvtColor(image_data_right, cv2.COLOR_BGRA2RGB)

            original_size = (image_rgb_left.shape[1], image_rgb_left.shape[0])
            target_training_size = (1248, 384) # fix later (should be same as training size)
            padded_left, pad_info_left = self.letterbox_image(image_rgb_left, target_training_size)
            padded_right, pad_info_right = self.letterbox_image(image_rgb_right, target_training_size)

            tensor_left = self.transform(padded_left)
            tensor_right = self.transform(padded_right)
            input_batch = torch.stack([tensor_left, tensor_right]).to(self.device)
            
            with torch.no_grad():
                output_batch = self.model(input_batch)
            predicted_masks = torch.argmax(output_batch, dim=1).cpu().numpy()
            mask_left = predicted_masks[0]
            mask_right = predicted_masks[1]

            visual_mask_left = (mask_left * 255).astype(np.uint8)
            visual_mask_right = (mask_right * 255).astype(np.uint8)

            final_mask_left = self.unletterbox_mask(visual_mask_left, pad_info_left, original_size)
            final_mask_right = self.unletterbox_mask(visual_mask_right, pad_info_right, original_size)

            ros_image_left = self.bridge.cv2_to_imgmsg(final_mask_left, encoding="mono8")
            ros_image_right = self.bridge.cv2_to_imgmsg(final_mask_right, encoding="mono8")

            self.pub_left.publish(ros_image_left)
            self.pub_right.publish(ros_image_right)
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
