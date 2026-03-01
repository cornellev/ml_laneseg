import rclpy 
from rclpy.node import Node
import pyzed.sl as sl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 

class ZedCameraTestNode(Node):
    def __init__(self, name='zed_camera_test_node'):
        super().__init__(name)

        # 1. Initialize ZED Camera
        self.zed = sl.Camera()

        # 2. Config parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 25

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f'Failed to open ZED camera: {err}')
            exit(-1)

        self.image_zed_left = sl.Mat()
        self.image_zed_right = sl.Mat()

        # 3. Setup ROS 2 Publishers for Raw Images
        self.pub_left = self.create_publisher(Image, 'zed/left/image_raw', 10)
        self.pub_right = self.create_publisher(Image, 'zed/right/image_raw', 10)
        self.bridge = CvBridge()

        # 4. Timer (Simulate 25 FPS)
        timer_period = 1.0 / 25.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('ZED Camera Test Node Started Successfully')

    def timer_callback(self):
        # Grab a new frame from the camera
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            
            # Retrieve images from the ZED API
            self.zed.retrieve_image(self.image_zed_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_zed_right, sl.VIEW.RIGHT)

            # Convert to numpy arrays (ZED outputs BGRA format)
            image_data_left = self.image_zed_left.get_data() 
            image_data_right = self.image_zed_right.get_data() 

            # Convert BGRA to BGR for standard ROS 2 viewing
            image_bgr_left = cv2.cvtColor(image_data_left, cv2.COLOR_BGRA2BGR)
            image_bgr_right = cv2.cvtColor(image_data_right, cv2.COLOR_BGRA2BGR)

            # Convert OpenCV images to ROS 2 Image messages
            ros_image_left = self.bridge.cv2_to_imgmsg(image_bgr_left, encoding="bgr8")
            ros_image_right = self.bridge.cv2_to_imgmsg(image_bgr_right, encoding="bgr8")

            # Publish the images
            self.pub_left.publish(ros_image_left)
            self.pub_right.publish(ros_image_right)
            
            # Throttled logging so it doesn't spam your terminal 25 times a second
            self.get_logger().info('Publishing raw camera frames...', throttle_duration_sec=2.0)
        
def main(args=None):
    rclpy.init(args=args)
    node = ZedCameraTestNode()
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