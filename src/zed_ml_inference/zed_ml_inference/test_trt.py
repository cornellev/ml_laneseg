import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms

class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Explicitly maintain the CUDA context for the Jetson GPU
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
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = trt.nptype(dtype_trt)
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

    def __call__(self, input_tensor):
        # Push the context before inference to avoid "invalid resource handle"
        self.cfx.push()
        
        cuda.memcpy_dtod_async(self.input_ptr, input_tensor.data_ptr(), 
                               input_tensor.element_size() * input_tensor.nelement(), self.stream)
        
        # Use execute_async_v3 for modern TRT versions
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        out_tensor = torch.empty(self.out_shape, dtype=torch.float16, device=input_tensor.device)
        cuda.memcpy_dtod_async(out_tensor.data_ptr(), self.output_ptr, 
                               out_tensor.element_size() * out_tensor.nelement(), self.stream)
        
        self.stream.synchronize()
        self.cfx.pop() # Remove context after use
        return out_tensor

    def __del__(self):
        self.cfx.pop()

def main():
    engine_path = '/ros2_ws/src/zed_ml_inference/zed_ml_inference/lfd_roadseg.engine'
    input_path = 'test_input.png'
    output_path = 'test_output_mask.png'
    device = torch.device('cuda:0')

    print(f"Loading Engine: {engine_path}")
    model = TRTEngine(engine_path)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not find {input_path}")
        return

    # Pre-processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_resized = cv2.resize(img_rgb, (624, 192))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3598, 0.3653, 0.3662], std=[0.2573, 0.2663, 0.2756])
    ])
    
    # Ensure tensor is FP16 to match the engine
    input_tensor = transform(input_resized).unsqueeze(0).to(device).half()

    print("Running Inference...")
    try:
        output = model(input_tensor)
        
        # Post-processing
        # output is likely (1, 2, 192, 624) -> channel 0 is background, channel 1 is lane
        mask = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Convert to visible image (0 -> 0, 1 -> 255)
        mask_visual = (mask * 255).astype(np.uint8)
        
        # Optional: Apply a threshold if the argmax is noisy
        cv2.imwrite(output_path, mask_visual)
        print(f"Saved result to: {output_path}")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
