#!/bin/bash
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export GDK_BACKEND=wayland
LOG_FILE=/tmp/corrosion_two_stage_rt_live.log

echo Fu123456 | sudo -S fuser -k /dev/video20 >/dev/null 2>&1
sleep 1

cd /mnt/ssd/spacemit_project || exit 1
: > "$LOG_FILE"
stdbuf -oL -eL .venv/bin/python -u -m app.main \
  --config /mnt/ssd/spacemit_project/configs/vision_usb_corrosion_two_stage_rt.yaml \
  vision-stream \
  --backend usb_v4l2 \
  --interval-seconds 0.0 \
  --display-competition \
  2>&1 | tee "$LOG_FILE" | grep --line-buffered -v '^\[VisionService\]\[Timing\]'
