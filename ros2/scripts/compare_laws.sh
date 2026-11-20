#!/usr/bin/env bash
# Run the through-singularity trajectory on the Webots arm once per control law,
# writing one CSV per law into /out:
#
#   docker run --rm --shm-size=1g -v "$PWD/results/webots:/out" \
#     --entrypoint bash singularity-arm-webots \
#     /opt/ws/src/singularity_robust_control/ros2/scripts/compare_laws.sh
#
# Each law gets a fresh launch so all three start from the same configuration.
# No `set -u`: ROS's setup.bash reads an unset variable.
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

  # loop:false: run the trajectory once, like the offline runs.
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
