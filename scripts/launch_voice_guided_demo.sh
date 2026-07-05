#!/bin/bash
set -euo pipefail

export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export GDK_BACKEND=wayland
export MULTIMODAL_TUI_TTY=/dev/tty
export SPACEMIT_MAIN_CPUSET=${SPACEMIT_MAIN_CPUSET:-0-7}
export SPACEMIT_VOICE_CPUSET=${SPACEMIT_VOICE_CPUSET:-4-7}
export SPACEMIT_TTS_CPUSET=${SPACEMIT_TTS_CPUSET:-0-3}
export SPACEMIT_ASR_CPUSET=${SPACEMIT_ASR_CPUSET:-0-3}
export SPACEMIT_VISION_CPUSET=${SPACEMIT_VISION_CPUSET:-0-3}
export SPACEMIT_OLLAMA_CPUSET=${SPACEMIT_OLLAMA_CPUSET:-${SPACEMIT_VOICE_CPUSET}}
export SPACEMIT_COMPETITION_NATIVE_VIDEO=${SPACEMIT_COMPETITION_NATIVE_VIDEO:-1}
export SPACEMIT_COMPETITION_DISPLAY_FPS=${SPACEMIT_COMPETITION_DISPLAY_FPS:-30}
MULTIMODAL_CPUSET=${MULTIMODAL_CPUSET:-${SPACEMIT_MAIN_CPUSET}}
LAUNCH_TTY=$(tty 2>/dev/null || true)

LOG_FILE=/tmp/voice_guided_demo.log
OLLAMA_AFFINITY_WATCH_PID=""

sudo_run() {
  echo Fu123456 | sudo -S "$@" >/dev/null 2>&1
}

apply_ollama_affinity() {
  if [ -z "${SPACEMIT_OLLAMA_CPUSET}" ]; then
    return
  fi
  if ! command -v pgrep >/dev/null 2>&1; then
    return
  fi

  for pattern in "ollama serve" "ollama runner"; do
    while IFS= read -r pid; do
      [ -n "${pid}" ] || continue
      sudo_run taskset -cp "${SPACEMIT_OLLAMA_CPUSET}" "${pid}" || true
    done < <(pgrep -u ollama -f "${pattern}" || true)
  done
}

start_ollama_affinity_watcher() {
  if [ -z "${OLLAMA_AFFINITY_WATCH_PID}" ] && [ -n "${SPACEMIT_OLLAMA_CPUSET}" ]; then
    (
      while true; do
        apply_ollama_affinity
        sleep 1
      done
    ) &
    OLLAMA_AFFINITY_WATCH_PID=$!
  fi
}

cleanup() {
  if [ -n "${OLLAMA_AFFINITY_WATCH_PID}" ]; then
    kill "${OLLAMA_AFFINITY_WATCH_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

sudo_run -v || true
sudo_run fuser -k /dev/video20 || true
apply_ollama_affinity
start_ollama_affinity_watcher
sleep 1

cd /mnt/ssd/spacemit_project || exit 1
: > "$LOG_FILE"

if [ -n "${LAUNCH_TTY}" ]; then
  {
    echo
    echo "[launch] voice-guided demo is starting, warmup may take about 30-40 seconds."
    echo "[launch] operator TUI will attach to ${MULTIMODAL_TUI_TTY} after startup."
    echo "[launch] runtime log: ${LOG_FILE}"
    echo
  } > "${LAUNCH_TTY}" 2>/dev/null || true
fi

exec > >(stdbuf -oL -eL tee -a "$LOG_FILE" >/dev/null) 2>&1

echo "[launch] MULTIMODAL_CPUSET=${MULTIMODAL_CPUSET}"
echo "[launch] SPACEMIT_VOICE_CPUSET=${SPACEMIT_VOICE_CPUSET}"
echo "[launch] SPACEMIT_TTS_CPUSET=${SPACEMIT_TTS_CPUSET}"
echo "[launch] SPACEMIT_ASR_CPUSET=${SPACEMIT_ASR_CPUSET}"
echo "[launch] SPACEMIT_VISION_CPUSET=${SPACEMIT_VISION_CPUSET}"
echo "[launch] SPACEMIT_OLLAMA_CPUSET=${SPACEMIT_OLLAMA_CPUSET}"
echo "[launch] SPACEMIT_COMPETITION_NATIVE_VIDEO=${SPACEMIT_COMPETITION_NATIVE_VIDEO}"
echo "[launch] SPACEMIT_COMPETITION_DISPLAY_FPS=${SPACEMIT_COMPETITION_DISPLAY_FPS}"
echo "[launch] log file: ${LOG_FILE}"

run_demo() {
  if command -v taskset >/dev/null 2>&1 && [ -n "${MULTIMODAL_CPUSET}" ]; then
    taskset -c "${MULTIMODAL_CPUSET}" \
      stdbuf -oL -eL .venv/bin/python -u -m app.main \
        --config /mnt/ssd/spacemit_project/configs/voice_guided_demo.yaml \
        voice-guided-demo
  else
    stdbuf -oL -eL .venv/bin/python -u -m app.main \
      --config /mnt/ssd/spacemit_project/configs/voice_guided_demo.yaml \
      voice-guided-demo
  fi
}

if ! run_demo; then
  if [ -n "${LAUNCH_TTY}" ]; then
    {
      echo
      echo "[launch] voice-guided demo exited unexpectedly, tailing ${LOG_FILE}:"
      tail -n 60 "${LOG_FILE}"
      echo
    } > "${LAUNCH_TTY}" 2>/dev/null || true
  fi
  exit 1
fi
