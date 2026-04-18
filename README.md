# ml_laneseg
How to run:

git clone https://github.com/cornellev/ml_laneseg.git

docker build -t lane_seg -f Dockerfile_cuda.jetson .

docker run -it --rm \
    --runtime nvidia \
    --privileged \
    --network host \
    -v /dev:/dev \
    -v $(pwd):/ros2_ws \
    -v ~/zed_models:/usr/local/zed/resources \
    lane_seg bash

wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y cudss && \
    rm cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
    
pip install pycuda
pip install onnxscript 

export FASTRTPS_DEFAULT_PROFILES_FILE=/ros2_ws/shm_disable.xml

colcon build --packages-select zed_ml_inference --symlink-install 
source install/setup.bash

ros2 run zed_ml_inference lane_segmentation_node3dFAST
