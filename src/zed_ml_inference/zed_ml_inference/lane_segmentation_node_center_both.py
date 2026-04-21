import rclpy 
from rclpy.node import Node
import pyzed.sl as sl
import torch 
import torch.nn.functional as F
import sys
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
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class DualLaneSegmentationNode3D(Node):
    def __init__(self):
        super().__init__('dual_lane_segmentation_node')
        
        # --- ZED 2 INITIALIZATION (Main, Forward) ---
        self.zed2 = sl.Camera()
        init_2 = sl.InitParameters()
        init_2.set_from_camera_id(0)
        init_2.camera_resolution = sl.RESOLUTION.VGA
        init_2.camera_fps = 15 # Hardware cap
        # CRITICAL: Lowest memory footprint possible
        init_2.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
        init_2.coordinate_units = sl.UNIT.METER

        if self.zed2.open(init_2) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("ZED 2 Open Failed")
            exit(-1)
            
        # --- ZED 1 INITIALIZATION (Secondary, Angled) ---
        self.zed1 = sl.Camera()
        init_1 = sl.InitParameters()
        init_1.set_from_camera_id(1)
        init_1.camera_resolution = sl.RESOLUTION.VGA
        init_1.camera_fps = 15
        init_1.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
        init_1.coordinate_units = sl.UNIT.METER

        if self.zed1.open(init_1) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().warn("ZED 1 Open Failed. Running in Single Camera Mode.")
            self.has_zed1 = False
        else:
            self.has_zed1 = True
            
        self.image_zed = sl.Mat()
        self.pc_zed = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        
        # --- PYTORCH MODEL INITIALIZATION ---
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cfg = {'scale_factor': 2}
        self.model = LFD_RoadSeg(**self.cfg)
        
        weights_path = '/ros2_ws/src/zed_ml_inference/zed_ml_inference/model_epoch_150.pth'
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).half().eval()

        self.edge_kernel = torch.tensor([[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]], device=self.device, dtype=torch.float16).unsqueeze(0)
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
        ])

        # --- ROS PUBLISHERS ---
        self.pub_overlay_wide = self.create_publisher(Image, 'lane_overlay_wide', 10)
        self.pub_marker_bounds = self.create_publisher(MarkerArray, 'lane_markers_3d', 10)
        self.pub_marker_center = self.create_publisher(MarkerArray, 'lane_centerline_3d', 10)
        self.bridge = CvBridge()
        
        # ROS 2 Timer running slightly slower than hardware to prevent buffer bloat
        self.create_timer(1.0/12.0, self.timer_callback)
        self.get_logger().info("Lightweight Dual Camera Node Started.")

    def create_marker(self, points, frame_id, r, g, b, ns, m_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns; m.id = m_id; m.type = Marker.POINTS; m.action = Marker.ADD
        m.scale.x = m.scale.y = 0.05
        m.color.a = 1.0; m.color.r = float(r); m.color.g = float(g); m.color.b = float(b)
        
        step = max(1, len(points) // 200) 
        for i in range(0, len(points), step):
            p = Point(); p.x, p.y, p.z = float(points[i][0]), float(points[i][1]), float(points[i][2])
            m.points.append(p)
        return m

    def process_camera(self, zed_cam, frame_id, offset):
        if zed_cam.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None, None, None

        zed_cam.retrieve_image(self.image_zed, sl.VIEW.LEFT)
        zed_cam.retrieve_measure(self.pc_zed, sl.MEASURE.XYZ)
        
        img_bgr = cv2.cvtColor(self.image_zed.get_data(), cv2.COLOR_BGRA2BGR)
        img_rgb = cv2.cvtColor(self.image_zed.get_data(), cv2.COLOR_BGRA2RGB)
        h, w = img_bgr.shape[:2]
        
        tensor_in = self.transform(cv2.resize(img_rgb, (624, 192))).unsqueeze(0).to(self.device).half()
        
        with torch.no_grad():
            output = self.model({"img": tensor_in})
            if isinstance(output, (list, tuple)): output = output[0]
        
        mask = torch.argmax(output, dim=1).half()
        
        edges = F.conv2d(mask.unsqueeze(0), self.edge_kernel, padding=1).squeeze()
        edges_full = F.interpolate(edges.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze()
        ys_b, xs_b = torch.where(edges_full > 0.1)
        ys_b, xs_b = ys_b.cpu().numpy(), xs_b.cpu().numpy()

        mask_full = F.interpolate(mask.unsqueeze(0).float(), size=(h, w), mode='nearest').squeeze().cpu().numpy()
        y_coords = np.arange(h); x_coords = np.arange(w)
        mass = mask_full.sum(axis=1) 
        valid_rows = mass > 0 
        ys_c = y_coords[valid_rows]
        xs_c = ((mask_full[valid_rows] * x_coords).sum(axis=1) / mass[valid_rows]).astype(int)

        pc_data = self.pc_zed.get_data()
        m_bounds = None; m_center = None

        if len(ys_b) > 0:
            pts_b = pc_data[ys_b, xs_b]
            pts_b = pts_b[~np.isnan(pts_b[:, 2]) & np.isfinite(pts_b[:, 2])]
            if len(pts_b) > 0:
                m_bounds = self.create_marker(pts_b, frame_id, 1.0, 0.0, 0.0, "bounds", offset)

        if len(ys_c) > 0:
            pts_c = pc_data[ys_c, xs_c]
            pts_c = pts_c[~np.isnan(pts_c[:, 2]) & np.isfinite(pts_c[:, 2])]
            if len(pts_c) > 0:
                g = 1.0 if offset == 0 else 0.0
                b = 0.0 if offset == 0 else 1.0
                m_center = self.create_marker(pts_c, frame_id, 0.0, g, b, "center", offset + 1)

        img_bgr[ys_b, xs_b] = [0, 0, 255] 
        img_bgr[ys_c, xs_c] = [0, 255, 0] 

        return m_bounds, m_center, img_bgr

    def timer_callback(self):
        arr_b = MarkerArray(); arr_c = MarkerArray()

        b2, c2, img2 = self.process_camera(self.zed2, "zed2_left_camera_frame", 0)
        if b2: arr_b.markers.append(b2)
        if c2: arr_c.markers.append(c2)

        final_img = img2

        if self.has_zed1:
            b1, c1, img1 = self.process_camera(self.zed1, "zed1_left_camera_frame", 2)
            if b1: arr_b.markers.append(b1)
            if c1: arr_c.markers.append(c1)
            if img1 is not None and img2 is not None:
                final_img = cv2.hconcat([img2, img1])

        if arr_b.markers: self.pub_marker_bounds.publish(arr_b)
        if arr_c.markers: self.pub_marker_center.publish(arr_c)
        if final_img is not None:
            self.pub_overlay_wide.publish(self.bridge.cv2_to_imgmsg(final_img, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = DualLaneSegmentationNode3D()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        try: 
            node.zed2.close(); 
            if node.has_zed1: node.zed1.close()
        except: pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()