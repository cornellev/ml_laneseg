import pyzed.sl as sl
import cv2

def main():
    print("Initializing ZED Camera...")
    zed = sl.Camera()

    # Set basic configuration
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"FAILED TO OPEN CAMERA: {err}")
        exit(1)

    print("Camera opened successfully! Grabbing frame...")
    image_mat = sl.Mat()
    
    # Grab a single frame
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        
        # Extract numpy array and save directly to the mounted folder
        image_data = image_mat.get_data()
        cv2.imwrite('/ros2_ws/sdk_success_test.jpg', image_data)
        
        print("✅ SUCCESS! Frame grabbed and saved as 'sdk_success_test.jpg'")
    else:
        print("❌ FAILED to grab frame from the camera.")

    zed.close()

if __name__ == "__main__":
    main()
