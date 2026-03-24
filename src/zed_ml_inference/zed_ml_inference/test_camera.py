import rclpy 
from rclpy.node import Node
import pyzed.sl as sl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
import os # Added to verify file creation

class ZedCameraTestNode(Node):
    def __init__(self, name='zed_camera_test_node'):
        super().__init__(name)

        self.get_logger().info('--- STEP 1: Booting Node and Opening Camera... ---')
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30 
        init_params.depth_mode = sl.DEPTH_MODE.NONE

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f'Failed to open ZED camera: {err}')
            exit(-1)

        self.get_logger().info('--- STEP 2: Camera Opened Successfully! ---')

        self.image_zed_left = sl.Mat()
        self.image_zed_right = sl.Mat()

        self.pub_left = self.create_publisher(Image, 'zed/left/image_raw', 10)
        self.pub_right = self.create_publisher(Image, 'zed/right/image_raw', 10)
        self.bridge = CvBridge()

        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('--- STEP 3: Setup Complete. Waiting for timer callback... ---')

    def timer_callback(self):
        self.get_logger().info('--- STEP 4: Timer Fired! Attempting to grab frame... ---', throttle_duration_sec=2.0)
        
        grab_state = self.zed.grab()
        if grab_state == sl.ERROR_CODE.SUCCESS:
            
            self.zed.retrieve_image(self.image_zed_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_zed_right, sl.VIEW.RIGHT)

            image_data_left = self.image_zed_left.get_data() 
            image_data_right = self.image_zed_right.get_data() 

            # DEBUG: Print the shape of the array to prove it isn't empty
            self.get_logger().info(f'--- STEP 5: Array extracted. Shape is: {image_data_left.shape} ---', throttle_duration_sec=2.0)

            image_bgr_left = cv2.cvtColor(image_data_left, cv2.COLOR_BGRA2BGR)
            image_bgr_right = cv2.cvtColor(image_data_right, cv2.COLOR_BGRA2BGR)

            ros_image_left = self.bridge.cv2_to_imgmsg(image_bgr_left, encoding="bgr8")
            ros_image_right = self.bridge.cv2_to_imgmsg(image_bgr_right, encoding="bgr8")

            current_time = self.get_clock().now().to_msg()
            ros_image_left.header.stamp = current_time
            ros_image_left.header.frame_id = 'zed_camera'
            ros_image_right.header.stamp = current_time
            ros_image_right.header.frame_id = 'zed_camera'

            # DEBUG: Force OpenCV to tell us if it succeeded
            save_path = '/ros2_ws/ros_node_success.jpg'
            self.get_logger().info('--- STEP 6: Attempting cv2.imwrite... ---', throttle_duration_sec=2.0)
            write_success = cv2.imwrite(save_path, image_bgr_left)
            self.get_logger().info(f'--- STEP 7: cv2.imwrite returned: {write_success} ---', throttle_duration_sec=2.0)

            # DEBUG: Force Linux to check if the file actually exists
            if os.path.exists(save_path):
                self.get_logger().info(f'--- STEP 8: SUCCESS! File physically exists at {save_path} ---', throttle_duration_sec=2.0)
            else:
                self.get_logger().error(f'--- STEP 8: GHOST FAILURE! File is missing from {save_path} ---', throttle_duration_sec=2.0)

            self.pub_left.publish(ros_image_left)
            self.pub_right.publish(ros_image_right)
            
        else:
            self.get_logger().error(f'--- FAILED to grab frame. ZED Error Code: {grab_state} ---', throttle_duration_sec=2.0)
        
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
