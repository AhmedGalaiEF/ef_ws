FROM theconstructai/ros-humble-beyondmimic:arm64

RUN cd /colcon_ws/src && \
    git clone https://bitbucket.org/theconstructcore/g1_ros2_control.git && \
    cd /colcon_ws && \
    bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select g1_ros2_control"

# Setup entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Source ROS 2 setup\n\
source /opt/ros/humble/setup.bash\n\
\n\
# Source workspace setup if it exists\n\
if [ -f "/colcon_ws/install/setup.bash" ]; then\n\
    source /colcon_ws/install/setup.bash\n\
fi\n\
\n\
exec "$@"' > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
