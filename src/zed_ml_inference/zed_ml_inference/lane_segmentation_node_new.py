import rclpy 
from rclpy.node import Node
import pyzed.sl as sl
import torch 
import sys
import time
from torchvision import transforms
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
import numpy as np 
import os 

# --- ORIGINAL MODEL IMPORT LOGIC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
LFD_REPO_PATH = "/ros2_ws/src/LFD_RoadSeg"

if os.path.exists(LFD_REPO_PATH):
    sys.path.append(LFD_REPO_PATH)

try:
    from models._LFDRoadSeg import LFD_RoadSeg
    print("Success: LFD_RoadSeg imported.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class LaneSegmentationNode(Node):
    def __init__(self, name='lane_segmentation_node'):
        super().__init__(name)
        
        # Performance Monitoring
        self.start_time = time.time()
        self.frame_counter = 0

        self.zed = sl.Camera()

        # --- ORIGINAL ZED INIT ---
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps = 30 
        init_params.depth_mode = sl.DEPTH_MODE.NONE 

        err = self.zed.open(init_params)
        if (err != sl.ERROR_CODE.SUCCESS) :
            exit(-1)

        self.image_zed_left = sl.Mat()
        timer_period = 1.0 / 30.0  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.cfg = {'scale_factor': 2}
        
        # --- ORIGINAL MODEL LOADING ---
        self.get_logger().info('Loading custom LFD_RoadSeg Model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LFD_RoadSeg(**self.cfg)
        weights_path = 'model_epoch_150.pth'
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device).half().eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
        ])

        # Publishers
        self.pub_mask = self.create_publisher(Image, 'lane_mask/left', 10)
        self.pub_overlay = self.create_publisher(Image, 'lane_overlay/left', 10)
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
            image_data = self.image_zed_left.get_data() 

            # Prepare images
            image_bgr = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
            image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
            original_size = (image_rgb.shape[1], image_rgb.shape[0])

            # AI Inference
            padded, pad_info = self.letterbox_image(image_rgb)
            tensor_in = self.transform(padded).unsqueeze(0).to(self.device).half()
            
            with torch.no_grad():
                output = self.model({"img": tensor_in})
                if isinstance(output, (list, tuple)):
                    output = output[0]
                    
            mask_idx = torch.argmax(output, dim=1).cpu().numpy()[0]
            visual_mask = (mask_idx * 255).astype(np.uint8)
            final_mask = self.unletterbox_mask(visual_mask, pad_info, original_size)

            # --- NEW RED OVERLAY LOGIC ---
            overlay = image_bgr.copy()
            # Apply red tint where final_mask is white (lane detected)
            overlay[final_mask > 127] = overlay[final_mask > 127] * 0.5 + np.array([0, 0, 255]) * 0.5

            # Convert and Publish
            ros_mask = self.bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
            ros_overlay = self.bridge.cv2_to_imgmsg(overlay.astype(np.uint8), encoding="bgr8")

            current_time = self.get_clock().now().to_msg()
            for msg in [ros_mask, ros_overlay]:
                msg.header.stamp = current_time
                msg.header.frame_id = 'zed_camera'

            self.pub_mask.publish(ros_mask)
            self.pub_overlay.publish(ros_overlay)
            
            # --- NEW FREQUENCY MONITOR ---
            self.frame_counter += 1
            now = time.time()
            if now - self.start_time > 2.0:
                fps = self.frame_counter / (now - self.start_time)
                self.get_logger().info(f'--- PUBLISHING AT: {fps:.2f} FPS ---')
                self.frame_counter = 0
                self.start_time = now

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

if __name__ == '__main__':
    main()
