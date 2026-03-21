#!/bin/bash
cd ~/ml_laneseg
 
docker run -it --rm \
    --runtime nvidia \
    --privileged \
    --network host \
    -v /dev:/dev \
    -v /usr/local/zed/resources:/usr/local/zed/resources \
    -v $(pwd)/src:/ros2_ws/src \
    lane_segr bash
