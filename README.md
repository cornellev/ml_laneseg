# ml_laneseg
How to run:

git clone https://github.com/cornellev/ml_laneseg.git

PIXI INSTRUCTIONS:
(all from the mllaneseg folder itself)
pixi install
pixi run build 
pixi run run-node
pixi run start-foxglove

DOCKER INSTRUCTIONS:
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

ros2 run zed_ml_inference lane_segmentation_node3dlanes

FOR TESTING THE MODEL:
go to src/zed_ml_inference/zed_ml_inference
change code for the picture that you want to test on
jsut lfd roadseg byt iself: 
python3 test_model.py

for the upgraded version with multilane + intersection capabilities: 
python3 test_model_multilane.py
