# ZED SDK 5.2 + ROS2 Humble + PyTorch on Jetson L4T R36.4 (JetPack 6.1/6.2)
FROM stereolabs/zed:5.2-py-devel-l4t-r36.4

# ── Environment ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    ROS_ROOT=/opt/ros/humble \
    PYTHONDONTWRITEBYTECODE=1

# ── ROS2 Humble ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg2 lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-cv-bridge \
        ros-humble-sensor-msgs \
        ros-humble-nmea-msgs \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-pip \
    && rosdep init && rosdep update \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────────────
# Pin numpy<2: cv_bridge and other ROS packages were compiled against NumPy 1.x
# and will crash with the _ARRAY_API error on NumPy 2.x
RUN pip3 install --no-cache-dir \
        "numpy<2" \
        torchvision \
        opencv-python-headless \
        tqdm \
        PyYAML \
        openpyxl \
        thop \
        fonttools

# ── Copy workspace ───────────────────────────────────────────────────────────
WORKDIR /ros2_ws
COPY src/ src/

# LFD_RoadSeg has no __init__.py files or setup.py, so find_packages() finds nothing.
# Create them so `from models._LFDRoadSeg import LFD_RoadSeg` works anywhere.
RUN touch src/LFD_RoadSeg/__init__.py \
          src/LFD_RoadSeg/models/__init__.py \
          src/LFD_RoadSeg/models/loss/__init__.py \
          src/LFD_RoadSeg/models/net/__init__.py \
          src/LFD_RoadSeg/models/scheduler/__init__.py \
    && printf 'from setuptools import setup, find_packages\nsetup(name="LFD_RoadSeg", packages=find_packages())\n' \
        > src/LFD_RoadSeg/setup.py \
    && pip3 install --no-cache-dir -e src/LFD_RoadSeg/

# ── Copy model weights ───────────────────────────────────────────────────────
# Place model_epoch_150.pth next to the Dockerfile before building
COPY model_epoch_150.pth src/zed_ml_inference/zed_ml_inference/model_epoch_150.pth

# ── Build workspace ──────────────────────────────────────────────────────────
# zed_debug links against libcuda/NvBuf/Argus which only exist on the Jetson at
# runtime — skip it. Everything else builds fine at image-build time.
RUN apt-get update && /bin/bash -c "\
    source ${ROS_ROOT}/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install \
        --packages-skip zed_debug zed_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release" \
    && rm -rf /var/lib/apt/lists/*

# ── Entrypoint ───────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "zed_ml_inference", "lane_segmentation_node"]
