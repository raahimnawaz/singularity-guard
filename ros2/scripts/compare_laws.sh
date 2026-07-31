#!/usr/bin/env bash
# Phase 7: run the through-singularity trajectory on the Webots plant once per
# control law, and drop one CSV per law into /out.
#
#   docker run --rm --shm-size=1g -v "$PWD/results/webots:/out" \
#     --entrypoint bash singularity-arm-webots \
#     /opt/ws/src/singularity_robust_control/ros2/scripts/compare_laws.sh
#
# --shm-size=1g is required: Docker's default 64 MB /dev/shm is too small for
# FastDDS's shared-memory transport, and the failure mode is topics continuing
# to flow while services silently time out.
#
# Each law gets a fresh launch rather than a live parameter change. controller_law
# is read in on_configure, so switching it live would need an unload/reload -- and
# more importantly a relaunch is the only way to guarantee all three runs start
# from the same joint configuration. Comparing a run that began wherever the
# previous one stopped would not be a comparison of control laws.
# Deliberately no `set -u`: ROS's setup.bash dereferences AMENT_TRACE_SETUP_FILES
# before defining it, so `set -u` aborts the script on the very first source with
# "unbound variable" -- and because the launch output is redirected to a log, the
# failure is silent. Every variable below has an explicit default instead.
set -o pipefail

CFG=/opt/ws/install/armctl_ros2_control/share/armctl_ros2_control/config/controllers.yaml
OUT=${OUT:-/out}
DURATION=${DURATION:-2.0}
SETTLE=${SETTLE:-115}

mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash
source /opt/ws/install/setup.bash

for LAW in naive fixed adaptive; do
  echo "=============== $LAW ==============="

  # loop:false so the trajectory runs once and holds, matching the offline runs
  # in armsim rather than the continuous demo loop the default is tuned for.
  sed -i -E "s/^( *controller_law:).*/\1 ${LAW}/" "$CFG"
  sed -i -E "s/^( *loop:).*/\1 false/" "$CFG"
  grep -E "controller_law:|loop:" "$CFG"

  xvfb-run --auto-servernum --server-args="-screen 0 1280x720x24" \
    ros2 launch armctl_ros2_control webots_sim.launch.py mode:=fast \
    > "$OUT/${LAW}_launch.log" 2>&1 &
  LAUNCH_PID=$!

  sleep "$SETTLE"

  timeout 180 ros2 run armctl_ros2_control record_run.py \
    --out "$OUT/${LAW}.csv" --duration "$DURATION" \
    >> "$OUT/${LAW}_launch.log" 2>&1
  echo "recorder exit: $?"

  kill -INT "$LAUNCH_PID" 2>/dev/null
  sleep 8
  pkill -9 -f webots 2>/dev/null
  pkill -9 -f ros2_supervisor 2>/dev/null
  pkill -9 -f webots-controller 2>/dev/null
  sleep 4

  if [ -f "$OUT/${LAW}.csv" ]; then
    echo "$LAW rows: $(( $(wc -l < "$OUT/${LAW}.csv") - 1 ))"
  else
    echo "$LAW: NO CSV -- see $OUT/${LAW}_launch.log"
  fi
done

echo "=============== done ==============="
ls -la "$OUT"
