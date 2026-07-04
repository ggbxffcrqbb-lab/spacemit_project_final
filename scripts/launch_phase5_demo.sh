#!/bin/bash
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export GDK_BACKEND=wayland
echo Fu123456 | sudo -S fuser -k /dev/video20 2>/dev/null
sleep 1
cd /mnt/ssd/spacemit_project || exit 1
echo "START: $(date +%H:%M:%S)"
.venv/bin/python -u -m app.main --config /mnt/ssd/spacemit_project/configs/vision_usb_defect_exp.yaml vision-stream --backend usb_v4l2 --interval-seconds 0.0 --display-competition --performance-mode 2>/tmp/phase5_demo_live_stderr.log
echo "EXIT:$?"
echo "END: $(date +%H:%M:%S)"
