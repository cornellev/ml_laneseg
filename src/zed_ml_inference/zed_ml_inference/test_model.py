import rclpy 
from rclpy.node import Node
import torch 
import sys
from torchvision import transforms
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
import numpy as np 
import os 

# Set up absolute pathing 
current_dir = os.path.dirname(os.path.abspath(__file__))
LFD_REPO_PATH = "/ros2_ws/src/src/LFD_RoadSeg"

if os.path.exists(LFD_REPO_PATH):
    sys.path.append(LFD_REPO_PATH)

try:
    from models._LFDRoadSeg import LFD_RoadSeg
    print("Success: LFD_RoadSeg imported.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class MockLaneSegmentationNode(Node):
    def __init__(self, name='mock_lane_segmentation_node'):
        super().__init__(name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Running on device: {self.device}')

        # 1. FIXED CONFIGURATION UNPACKING
        # Flattened the config so it can be safely unpacked into the constructor
        self.cfg = {
            'scale_factor': 2
        }
        
        # The '**' unpacks the dictionary into kwargs (e.g., scale_factor=2)
        # This stops the model from absorbing the dictionary as a single variable
        self.model = LFD_RoadSeg(**self.cfg)

        weights_path = os.path.join(current_dir, 'model_epoch_150.pth')
        if not os.path.exists(weights_path):
            self.get_logger().error(f'Weights not found at {weights_path}')
            sys.exit(1)

        # 2. SILENCED THE SECURITY WARNING
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
        ])

        self.test_image_path = os.path.join(current_dir, "test_input.png")
        self.bridge = CvBridge()

        self.run_inference()

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
    
    def run_inference(self):
        image_data = cv2.imread(self.test_image_path)
        if image_data is None:
            self.get_logger().error(f'Could not read image: {self.test_image_path}')
            return

        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        original_size = (image_rgb.shape[1], image_rgb.shape[0])
        padded, pad_info = self.letterbox_image(image_rgb, (1248, 384))
        
        tensor = self.transform(padded).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 3. REQUIRED WRAPPER
            # The forward method explicitly expects a dictionary to pull "img" from
            model_input = {"img": tensor}
            output = self.model(model_input)
            
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        visual_mask = (mask * 255).astype(np.uint8)
        final_mask = self.unletterbox_mask(visual_mask, pad_info, original_size)
        
        # Create a blank colored mask (e.g., pure Green for the lane)
        color_mask = np.zeros_like(image_data)
        color_mask[:, :, 2] = final_mask  # Assign the white mask to the Green channel
        
        # Blend the original image and the green mask
        # 0.7 is the weight of the original image, 0.4 is the weight of the color
        overlay_image = cv2.addWeighted(image_data, 0.7, color_mask, 0.4, 0)
        
        output_path = os.path.join(current_dir, "output_debug.jpg")
        cv2.imwrite(output_path, overlay_image)
        print(f"Success! Saved overlay debug output to: {output_path}")

def main(args=None):
    rclpy.init(args=args)
    node = MockLaneSegmentationNode()
    node.destroy_node()
    rclpy.shutdown()
    print("Test finished.")

if __name__ == '__main__':
    main()