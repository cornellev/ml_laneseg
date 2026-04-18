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

# --- TENSORRT & CUDA ---
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        # Manually manage the CUDA context for stable Jetson performance
        self.cfx = cuda.Device(0).make_context()
        
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.engine.get_tensor_shape(name)
            dtype_np = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            itemsize = np.dtype(dtype_np).itemsize 
            device_mem = cuda.mem_alloc(size * itemsize)
            self.allocations.append(int(device_mem))
            self.context.set_tensor_address(name, int(device_mem))
            if is_input:
                self.input_ptr = device_mem
            else:
                self.output_ptr = device_mem
                self.out_shape = tuple(shape)
        self.cfx.pop()

    def __call__(self, input_tensor):
        self.cfx.push() 
        cuda.memcpy_dtod_async(self.input_ptr, input_tensor.data_ptr(), input_tensor.element_size() * input_tensor.nelement(), self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        out_tensor = torch.empty(self.out_shape, dtype=torch.float16, device=input_tensor.device)
        cuda.memcpy_dtod_async(out_tensor.data_ptr(), self.output_ptr, out_tensor.element_size() * out_tensor.nelement(), self.stream)
        self.stream.synchronize()
        self.cfx.pop() 
        return out_tensor

class LaneSegmentationNode3D(Node):
    def __init__(self):
        super().__init__('lane_segmentation_node')
        
        # --- ZED INITIALIZATION ---
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
        init_params.coordinate_units = sl.UNIT.METER
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("ZED Open Failed")
            exit(-1)
        
        self.image_zed = sl.Mat()
        self.pc_zed = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        
        # --- MODEL INITIALIZATION ---
        self.device = torch.device('cuda:0')
        self.trt_model = TRTEngine('/ros2_ws/src/zed_ml_inference/zed_ml_inference/lfd_roadseg.engine')
        
        self.edge_kernel = torch.tensor([[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]], device=self.device, dtype=torch.float16).unsqueeze(0)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
        ])

        # --- ROS PUBLISHERS (Fixed Topic Names) ---
        self.pub_overlay = self.create_publisher(Image, 'lane_overlay_left', 10)
        self.pub_marker = self.create_publisher(MarkerArray, 'lane_markers_3d', 10)
        self.bridge = CvBridge()
        
        self.create_timer(1.0/30.0, self.timer_callback)
        self.get_logger().info("FAST Lane Node Started. Compliant Topic names active.")

    def timer_callback(self):
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.pc_zed, sl.MEASURE.XYZ)
            
            img_full = self.image_zed.get_data()
            img_bgr = cv2.cvtColor(img_full, cv2.COLOR_BGRA2BGR)
            h, w = img_bgr.shape[:2]
            
            input_resized = cv2.resize(img_bgr, (624, 192))
            tensor_in = self.transform(input_resized).unsqueeze(0).to(self.device).half()
            output = self.trt_model(tensor_in)
            
            mask = torch.argmax(output, dim=1).half()
            edges = F.conv2d(mask.unsqueeze(0), self.edge_kernel, padding=1).squeeze()
            edges_full = F.interpolate(edges.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze()
            
            ys_t, xs_t = torch.where(edges_full > 0.1)
            ys, xs = ys_t.cpu().numpy(), xs_t.cpu().numpy()

            if len(ys) > 0:
                pc_data = self.pc_zed.get_data()
                points_3d = pc_data[ys, xs]
                valid_mask = ~np.isnan(points_3d[:, 2]) & np.isfinite(points_3d[:, 2])
                points_filtered = points_3d[valid_mask]

                if len(points_filtered) > 0:
                    marker_array = MarkerArray()
                    marker = Marker()
                    marker.header.frame_id = "zed_left_camera_frame"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.type = Marker.POINTS
                    marker.action = Marker.ADD
                    marker.scale.x = marker.scale.y = 0.04
                    marker.color.a, marker.color.r = 1.0, 1.0 
                    
                    step = max(1, len(points_filtered) // 200) 
                    for i in range(0, len(points_filtered), step):
                        p = Point()
                        p.x, p.y, p.z = float(points_filtered[i][0]), float(points_filtered[i][1]), float(points_filtered[i][2])
                        marker.points.append(p)
                    
                    marker_array.markers.append(marker)
                    self.pub_marker.publish(marker_array)

            img_bgr[ys, xs] = [0, 0, 255]
            self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(img_bgr, "bgr8"))

def main():
    rclpy.init()
    node = LaneSegmentationNode3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.zed.close()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
