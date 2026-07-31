# Build from the repo root:  docker build -f ros2/Dockerfile -t singularity-arm-webots .
#
# Bakes ROS2 Jazzy + webots_ros2_control + a prebuilt Webots R2025a into the
# image, so `docker run` doesn't have to download ~1.5 GB from GitHub on
# first launch (webots_ros2_driver's default behavior) or depend on network
# access at all after build time.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-webots-ros2-control \
      ros-jazzy-webots-ros2-driver \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-controller-manager \
      ros-jazzy-joint-state-broadcaster \
      ros-jazzy-robot-state-publisher \
      python3-colcon-common-extensions \
      xvfb \
      wget ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Must match webots_ros2_driver.utils.WebotsVersion.minimum() for this
# ros-jazzy-webots-ros2-control release. Asserted a few layers down, once the
# download this pin controls has already been cached.
ARG WEBOTS_VERSION=R2025a
RUN mkdir -p /root/.ros/webots${WEBOTS_VERSION} \
    && wget -q "https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/webots-${WEBOTS_VERSION}-x86-64.tar.bz2" \
      -O /tmp/webots.tar.bz2 \
    && tar xjf /tmp/webots.tar.bz2 -C /root/.ros/webots${WEBOTS_VERSION} \
    && rm /tmp/webots.tar.bz2
ENV WEBOTS_HOME=/root/.ros/webotsR2025a/webots

# Runtime deps for the Webots binary. ros:jazzy-ros-base is headless and ships
# almost none of these; without them webots-bin exits 127 on a missing .so
# before it ever opens the world, which surfaces downstream as the far more
# confusing "controller_manager never appears".
#
# Finding the list one failure per rebuild is slow, and a bare
# `ldd webots-bin | grep "not found"` over-reports badly: Webots bundles its own
# Qt6/ODE/assimp/OIS under lib/webots and resolves them via the LD_LIBRARY_PATH
# its launcher script sets, which ldd does not know about. Set that path first
# and the output becomes exactly the system libraries that are genuinely absent:
#
#   export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/webots:$LD_LIBRARY_PATH
#   ldd $WEBOTS_HOME/bin/webots-bin | grep "not found"
#   for f in $WEBOTS_HOME/lib/webots/qt/plugins/platforms/*.so; do \
#       ldd "$f" | grep "not found"; done
#
# The second loop matters: the Qt platform plugins are dlopen'd, so their
# dependencies never show up in the main binary's ldd output at all. With this
# list installed both commands report nothing missing. python3-pil is separate
# again -- it is for ros2_supervisor.py, which imports urdf2webots -> PIL.
#
# Deliberately a layer after the download above, so editing this list does not
# re-fetch ~1.5 GB of Webots.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglu1-mesa \
      libgl1 libglx-mesa0 libegl1 libgles2 libopengl0 libglvnd0 \
      libxkbcommon-x11-0 \
      libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
      libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
      libxcomposite1 libxcursor1 libxdamage1 libxi6 libxrandr2 libxtst6 \
      libnss3 libasound2t64 libsndio7.0 \
      libwayland-cursor0 libwayland-egl1 \
      python3-pil \
    && rm -rf /var/lib/apt/lists/*

# webots-controller reads USER to name its IPC endpoint; ros:jazzy-ros-base
# leaves it unset, which prints an error and falls back to 'default' on every
# connection attempt.
ENV USER=root

# Put FastDDS on UDP only. Its shared-memory transport is the least reliable
# thing in this image: it fails with "open_and_lock_file failed" even against a
# 1 GB /dev/shm, and it fails *partially* -- topics keep flowing while services
# stop completing, so controller spawners hang and it looks like a
# controller_manager bug. See the XML for the full reasoning.
COPY ros2/config/fastdds_udp_only.xml /opt/fastdds_udp_only.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/fastdds_udp_only.xml

# webots_ros2_driver decides it is running under WSL from the kernel string
# alone:  def is_wsl(): return 'microsoft-standard' in uname().release
# Docker Desktop on Windows runs its Linux VM *on top of* WSL2, so every
# container inherits a `...-microsoft-standard-WSL2` kernel and is_wsl()
# returns True inside a plain Linux container. The driver then shells out to
# `wslpath` -- a Windows-side tool that does not exist here -- and the launch
# file raises FileNotFoundError before Webots is ever started, which shows up
# downstream as controller_manager never appearing.
#
# Genuine WSL always has wslpath on PATH and this VM never does, so requiring
# it makes the detection *correct* rather than merely disabled. No-op on a
# native Linux host, where the kernel string does not match to begin with.
# utils.py already imports shutil. Asserted at build time, not assumed.
RUN UTILS=/opt/ros/jazzy/lib/python3.12/site-packages/webots_ros2_driver/utils.py \
    && sed -i "s|return 'microsoft-standard' in uname().release|return 'microsoft-standard' in uname().release and shutil.which('wslpath') is not None|" "$UTILS" \
    && python3 -c "\
import sys; sys.path.insert(0, '/opt/ros/jazzy/lib/python3.12/site-packages'); \
from webots_ros2_driver.utils import is_wsl; \
assert is_wsl() is False, 'is_wsl() still True after patch'; \
print('patched: is_wsl() ->', is_wsl())"

# The WEBOTS_VERSION pinned above is only correct as long as it matches what
# this webots_ros2_driver release actually requires; a silent mismatch would
# show up as a confusing runtime failure long after the build. Assert it here
# rather than in the download layer, so re-checking costs nothing.
ARG WEBOTS_VERSION
RUN python3 -c "\
import sys; sys.path.insert(0, '/opt/ros/jazzy/lib/python3.12/site-packages'); \
from webots_ros2_driver.utils import WebotsVersion; \
m = WebotsVersion.minimum().short(); \
assert m == '${WEBOTS_VERSION}', \
    f'webots_ros2_driver wants {m}, image pins ${WEBOTS_VERSION}'; \
print('webots version pin OK:', m)"

# cpp/include/armctl is pulled in unmodified via a relative include path
# (ros2/CMakeLists.txt: ../cpp/include) -- both must be copied to keep that
# path valid.
WORKDIR /opt/ws/src/singularity_robust_control
COPY cpp ./cpp
COPY ros2 ./ros2

WORKDIR /opt/ws
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-select armctl_ros2_control \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN chmod +x /opt/ws/src/singularity_robust_control/ros2/entrypoint.sh
ENTRYPOINT ["/opt/ws/src/singularity_robust_control/ros2/entrypoint.sh"]
CMD ["mode:=fast"]
