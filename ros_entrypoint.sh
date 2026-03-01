#!/bin/bash
set -e

# Source ROS 2 Humble
source "/opt/ros/humble/setup.bash"

# Source your workspace if it exists
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source "/ros2_ws/install/setup.bash"
fi

exec "$@"