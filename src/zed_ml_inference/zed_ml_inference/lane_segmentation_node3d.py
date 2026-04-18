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

class LaneSegmentationNode3D(Node):
    def __init__(self, name='lane_segmentation_node'):
        super().__init__(name)
        
        self.start_time = time.time()
        self.frame_counter = 0

        # --- ZED 2 HARDWARE INIT ---
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps = 30 
        
        # 1. Use NVIDIA Hardware Acceleration for Depth
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
        init_params.coordinate_units = sl.UNIT.METER

        err = self.zed.open(init_params)
        if (err != sl.ERROR_CODE.SUCCESS) :
            self.get_logger().error("Failed to open ZED camera")
            exit(-1)

        # 2. Enable IMU & Positional Tracking for stable 3D coordinates
        tracking_params = sl.PositionalTrackingParameters()
        err = self.zed.enable_positional_tracking(tracking_params)
        if err == sl.ERROR_CODE.SUCCESS:
            self.get_logger().info("ZED 2 IMU and Positional Tracking Enabled!")

        # Containers
        self.image_zed_left = sl.Mat()
        self.point_cloud = sl.Mat() 
        self.runtime_params = sl.RuntimeParameters()
        
        timer_period = 1.0 / 30.0  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.cfg = {'scale_factor': 2}
        
        self.get_logger().info('Loading custom PyTorch Model...')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LFD_RoadSeg(**self.cfg)
        weights_path = 'model_epoch_150.pth'
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        # FP16 Precision for memory savings
        self.model.to(self.device).half().eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
        ])

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
        return cv2.resize(cropped_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    def timer_callback(self):
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            
            # 1. Retrieve Image and ZED 2 Point Cloud
            self.zed.retrieve_image(self.image_zed_left, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ) 
            
            image_data = self.image_zed_left.get_data() 
            pc_data = self.point_cloud.get_data() # Numpy array of [X, Y, Z, NaN]

            image_bgr = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
            image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
            original_size = (image_rgb.shape[1], image_rgb.shape[0])

            # 2. AI Inference
            padded, pad_info = self.letterbox_image(image_rgb)
            tensor_in = self.transform(padded).unsqueeze(0).to(self.device).half()
            
            with torch.no_grad():
                output = self.model({"img": tensor_in})
                if isinstance(output, (list, tuple)):
                    output = output[0]
                    
            mask_idx = torch.argmax(output, dim=1).cpu().numpy()[0]
            visual_mask = (mask_idx * 255).astype(np.uint8)
            final_mask = self.unletterbox_mask(visual_mask, pad_info, original_size)

            # 3. EXTRACT EDGES ONLY
            edges = cv2.Canny(final_mask, 100, 200)

            # 4. FAST 3D MAPPING (Edges Only)
            ys, xs = np.where(edges > 127)
            
            if len(ys) > 0:
                # Extract the 3D coordinates for ONLY the edge pixels
                edge_3d_points = pc_data[ys, xs]
                
                # Filter out bad depth data (where camera couldn't see)
                valid_points = edge_3d_points[~np.isnan(edge_3d_points[:, 2]) & np.isfinite(edge_3d_points[:, 2])]
                
                if len(valid_points) > 0:
                    # Log the closest part of the lane edge (lowest Z value)
                    closest_pt = valid_points[np.argmin(valid_points[:, 2])]
                    self.get_logger().info(f"Closest Lane Edge -> X: {closest_pt[0]:.2f}m, Y: {closest_pt[1]:.2f}m, Z: {closest_pt[2]:.2f}m", throttle_duration_sec=1.0)

            # 5. RED EDGE OVERLAY
            overlay = image_bgr.copy()
            # Paint only the thin edge lines red
            overlay[edges > 127] = [0, 0, 255] 

            # Publish
            ros_mask = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            ros_overlay = self.bridge.cv2_to_imgmsg(overlay.astype(np.uint8), encoding="bgr8")

            current_time = self.get_clock().now().to_msg()
            for msg in [ros_mask, ros_overlay]:
                msg.header.stamp = current_time
                msg.header.frame_id = 'zed_camera'

            self.pub_mask.publish(ros_mask)
            self.pub_overlay.publish(ros_overlay)
            
            # --- FREQUENCY MONITOR ---
            self.frame_counter += 1
            now = time.time()
            if now - self.start_time > 2.0:
                fps = self.frame_counter / (now - self.start_time)
                self.get_logger().info(f'--- PUBLISHING AT: {fps:.2f} FPS ---')
                self.frame_counter = 0
                self.start_time = now

def main(args=None):
    rclpy.init(args=args)
    node = LaneSegmentationNode3D()
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
