# ml_laneseg
How to run:

git clone https://github.com/cornellev/ml_laneseg.git

docker build -t lane_segr_final -f Dockerfile_cuda.jetson .

docker run -it --rm \
    --runtime nvidia \
    --privileged \
    --network host \
    -v /dev:/dev \
    -v $(pwd):/ros2_ws \
    -v ~/zed_models:/usr/local/zed/resources \
    lane_segr_final bash

colcon build --packages-select zed_ml_inference --symlink-install 
source install/setup.bash

ros2 run zed_ml_inference lane_segmentation_node3dcenter

# OR

ros2 run zed_ml_inference lane_segmentation_node3dfastcenter
