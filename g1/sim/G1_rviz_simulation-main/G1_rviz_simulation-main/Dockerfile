# Build stage
FROM ghcr.io/sloretz/ros:humble-ros-core AS build_stage

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    git \
    python3-rosdep \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the repositories for the project
WORKDIR /ros2_ws/src
RUN git clone https://bitbucket.org/theconstructcore/g1_description.git

# Use /bin/bash for the next RUN command
SHELL ["/bin/bash", "-c"]

# Build the workspace
WORKDIR /ros2_ws
RUN source /ros_entrypoint.sh && colcon build

# Clone and build unitree_ros2 message packages
WORKDIR /
RUN git clone https://github.com/unitreerobotics/unitree_ros2.git && cd unitree_ros2/cyclonedds_ws && source /ros_entrypoint.sh && colcon build

# Runtime stage
FROM ghcr.io/sloretz/ros:humble-ros-core

RUN apt-get update && apt-get install -y \
    ros-humble-realsense2-description \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher

# Copy the build artifacts from the build stage
COPY --from=build_stage /ros2_ws /ros2_ws
COPY --from=build_stage /unitree_ros2/cyclonedds_ws /unitree_ros2/cyclonedds_ws

# Set the entrypoint for the runtime container
ENTRYPOINT [ "/ros_entrypoint.sh" ]
CMD [ "bash" ]
