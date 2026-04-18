import rclpy 
from rclpy.node import Node
import pyzed.sl as sl
import torch 
import torch.nn.functional as F
import sys
import time
from torchvision import transforms
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2 
import numpy as np 
import os 

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
    def __init__(self):
        super().__init__('lane_segmentation_node')
        
        # --- ZED 2 HARDWARE INIT ---
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps = 30 
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
        init_params.coordinate_units = sl.UNIT.METER

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("ZED Open Failed")
            exit(-1)
            
        self.image_zed = sl.Mat()
        self.pc_zed = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        
        # --- PYTORCH MODEL INITIALIZATION ---
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cfg = {'scale_factor': 2}
        self.get_logger().info('Loading custom PyTorch Model Shell...')
        
        # 1. Instantiate the empty architecture
        self.model = LFD_RoadSeg(**self.cfg)
        
        # 2. Load the state_dict weights
        weights_path = '/ros2_ws/src/zed_ml_inference/zed_ml_inference/model_epoch_150.pth'
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        # 3. Optimize for Jetson (FP16 and Eval)
        self.model.to(self.device).half().eval()
        self.get_logger().info('PyTorch Model weights loaded successfully!')

        # --- GPU POST-PROCESSING VARS ---
        self.edge_kernel = torch.tensor([[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]], device=self.device, dtype=torch.float16).unsqueeze(0)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
        ])

        # --- ROS PUBLISHERS ---
        self.pub_overlay = self.create_publisher(Image, 'lane_overlay_left', 10)
        self.pub_marker_bounds = self.create_publisher(MarkerArray, 'lane_markers_3d', 10)
        self.pub_marker_center = self.create_publisher(MarkerArray, 'lane_centerline_3d', 10)
        self.bridge = CvBridge()
        
        self.create_timer(1.0/30.0, self.timer_callback)

    def create_marker(self, points_filtered, color_r, color_g, color_b, ns, id_offset=0):
        """Helper to generate ROS markers"""
        marker = Marker()
        marker.header.frame_id = "zed_left_camera_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = id_offset
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.05
        marker.color.a = 1.0
        marker.color.r = float(color_r)
        marker.color.g = float(color_g)
        marker.color.b = float(color_b)
        
        step = max(1, len(points_filtered) // 200) 
        for i in range(0, len(points_filtered), step):
            p = Point()
            p.x, p.y, p.z = float(points_filtered[i][0]), float(points_filtered[i][1]), float(points_filtered[i][2])
            marker.points.append(p)
        return marker

    def timer_callback(self):
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.pc_zed, sl.MEASURE.XYZ)
            
            img_full = self.image_zed.get_data()
            img_bgr = cv2.cvtColor(img_full, cv2.COLOR_BGRA2BGR)
            img_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGRA2RGB)
            h, w = img_bgr.shape[:2]
            
            # Fast Resize (Avoiding CPU letterboxing for speed)
            input_resized = cv2.resize(img_rgb, (624, 192))
            tensor_in = self.transform(input_resized).unsqueeze(0).to(self.device).half()
            
            # --- PYTORCH INFERENCE ---
            with torch.no_grad():
                # Dictionary input matching your original code
                output = self.model({"img": tensor_in})
                if isinstance(output, (list, tuple)):
                    output = output[0]
            
            # --- 1. GET LANE BOUNDARIES (GPU Edges) ---
            mask = torch.argmax(output, dim=1).half()
            edges = F.conv2d(mask.unsqueeze(0), self.edge_kernel, padding=1).squeeze()
            edges_full = F.interpolate(edges.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze()
            
            ys_bound, xs_bound = torch.where(edges_full > 0.1)
            ys_b, xs_b = ys_bound.cpu().numpy(), xs_bound.cpu().numpy()

            # --- 2. GET LANE CENTERLINE ---
            # Correctly dimensioned interpolation
            mask_full = F.interpolate(mask.unsqueeze(0).float(), size=(h, w), mode='nearest').squeeze().cpu().numpy()
            
            y_coords = np.arange(h)
            x_coords = np.arange(w)
            mass = mask_full.sum(axis=1) 
            valid_rows = mass > 0 
            
            ys_c = y_coords[valid_rows]
            xs_c = (mask_full[valid_rows] * x_coords).sum(axis=1) / mass[valid_rows]
            xs_c = xs_c.astype(int)

            pc_data = self.pc_zed.get_data()

            # --- 3. PUBLISH BOUNDARIES (RED) ---
            if len(ys_b) > 0:
                points_bound = pc_data[ys_b, xs_b]
                valid_mask_b = ~np.isnan(points_bound[:, 2]) & np.isfinite(points_bound[:, 2])
                points_bound_filtered = points_bound[valid_mask_b]

                if len(points_bound_filtered) > 0:
                    marker_array_b = MarkerArray()
                    marker_array_b.markers.append(self.create_marker(points_bound_filtered, 1.0, 0.0, 0.0, "bounds", 0))
                    self.pub_marker_bounds.publish(marker_array_b)

            # --- 4. PUBLISH CENTERLINE (GREEN) ---
            if len(ys_c) > 0:
                points_center = pc_data[ys_c, xs_c]
                valid_mask_c = ~np.isnan(points_center[:, 2]) & np.isfinite(points_center[:, 2])
                points_center_filtered = points_center[valid_mask_c]

                if len(points_center_filtered) > 0:
                    marker_array_c = MarkerArray()
                    marker_array_c.markers.append(self.create_marker(points_center_filtered, 0.0, 1.0, 0.0, "center", 1))
                    self.pub_marker_center.publish(marker_array_c)

            # --- 5. VISUAL OVERLAY ---
            img_bgr[ys_b, xs_b] = [0, 0, 255] 
            img_bgr[ys_c, xs_c] = [0, 255, 0] 
            self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(img_bgr, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = LaneSegmentationNode3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.zed.close()
        except: pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()