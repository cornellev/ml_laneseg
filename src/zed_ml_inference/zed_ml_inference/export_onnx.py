import torch
import torch.nn as nn
import os
import sys

# Ensure your model can be found
LFD_REPO_PATH = "/ros2_ws/src/LFD_RoadSeg"
if os.path.exists(LFD_REPO_PATH):
    sys.path.append(LFD_REPO_PATH)

from models._LFDRoadSeg import LFD_RoadSeg

# Wrapper to remove the dictionary input requirement for TensorRT
class TRT_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, img):
        return self.model({"img": img})

def main():
    device = torch.device('cuda')
    cfg = {'scale_factor': 2}
    
    # 1. Load original model
    print("Loading PyTorch weights...")
    base_model = LFD_RoadSeg(**cfg)
    checkpoint = torch.load('model_epoch_150.pth', map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_model.load_state_dict(checkpoint)
        
    # 2. Wrap it and convert to FP16
    model = TRT_Wrapper(base_model).to(device).half().eval()
    
    # 3. Create a dummy tensor matching our optimized 624x192 resolution
    dummy_input = torch.randn(1, 3, 192, 624, dtype=torch.float16, device=device)
    
    print("Exporting PyTorch model to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        "lfd_roadseg.onnx", 
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output']
    )
    print("Export Complete! File saved as lfd_roadseg.onnx")

if __name__ == '__main__':
    main()
