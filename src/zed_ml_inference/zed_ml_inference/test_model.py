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
LFD_REPO_PATH = "/ros2_ws/src/LFD_RoadSeg"

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

        self.cfg = {'scale_factor': 2}
        self.model = LFD_RoadSeg(**self.cfg)

        weights_path = os.path.join(current_dir, 'model_epoch_150.pth')
        if not os.path.exists(weights_path):
            self.get_logger().error(f'Weights not found at {weights_path}')
            sys.exit(1)

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        
        # EXACT LIVE NODE MECHANISM: FP16 and Eval mode
        self.model.to(self.device).half().eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
        ])

        self.test_image_path = os.path.join(current_dir, "test_input.png")
        self.run_inference()
    
    def run_inference(self):
        image_data = cv2.imread(self.test_image_path)
        if image_data is None:
            self.get_logger().error(f'Could not read image: {self.test_image_path}')
            return

        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]
        
        # EXACT LIVE NODE MECHANISM: Fast resize to 624x192
        input_resized = cv2.resize(image_rgb, (624, 192))
        
        # EXACT LIVE NODE MECHANISM: Convert to FP16 (.half()) to match model
        tensor = self.transform(input_resized).unsqueeze(0).to(self.device).half()

        with torch.no_grad():
            model_input = {"img": tensor}
            output = self.model(model_input)
            
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Convert mask to 0 and 255
        visual_mask = (mask * 255).astype(np.uint8)
        
        # Because we didn't letterbox, we can just stretch the mask back to the original image size
        final_mask = cv2.resize(visual_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        # Create a blank colored mask (e.g., pure Green for the lane)
        color_mask = np.zeros_like(image_data)
        color_mask[:, :, 1] = final_mask  # Assign the white mask to the Green channel (BGR format)
        
        # Blend the original image and the green mask
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