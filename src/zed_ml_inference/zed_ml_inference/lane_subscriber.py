import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class LaneSegSubscriber(Node):
    def __init__(self):
        super().__init__('lane_seg_subscriber')
        self.bridge = CvBridge()
        
        # Subscribe to the Raw Mask
        self.sub_mask = self.create_subscription(
            Image,
            'lane_mask/left',
            self.mask_callback,
            10)
            
        # Subscribe to the Colored Overlay
        self.sub_overlay = self.create_subscription(
            Image,
            'lane_overlay/left',
            self.overlay_callback,
            10)
            
        self.get_logger().info('--- Dual Subscriber Started. Listening for Mask and Overlay... ---')

    def mask_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            cv2.imwrite('/ros2_ws/lane_mask_suc1cess.jpg', cv_image)
            self.get_logger().info('Saved updated B&W Mask!', throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f'Failed to process mask: {e}')

    def overlay_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite('/ros2_ws/lane_overlay_su1ccess.jpg', cv_image)
            self.get_logger().info('Saved updated Colored Overlay!', throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f'Failed to process overlay: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LaneSegSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
