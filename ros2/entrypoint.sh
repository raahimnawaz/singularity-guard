#!/usr/bin/env bash
# HEADLESS=1 (default): runs Webots under Xvfb, no host display needed --
# works in CI or a plain `docker run`.
# HEADLESS=0: runs against a real display -- pass -e DISPLAY=$DISPLAY and
# -v /tmp/.X11-unix:/tmp/.X11-unix (Linux/WSLg) so the Webots GUI shows up.
set -e

source /opt/ros/jazzy/setup.bash
source /opt/ws/install/setup.bash

if [ "${HEADLESS:-1}" = "1" ]; then
  exec xvfb-run --auto-servernum --server-args="-screen 0 1280x720x24" \
    ros2 launch armctl_ros2_control webots_sim.launch.py "$@"
else
  exec ros2 launch armctl_ros2_control webots_sim.launch.py "$@"
fi
