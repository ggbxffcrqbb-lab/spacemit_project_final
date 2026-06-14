#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

all_required_files_exist() {
  local base_dir="$1"
  shift
  local rel
  for rel in "$@"; do
    if [ ! -f "${base_dir}/${rel}" ]; then
      return 1
    fi
  done
  return 0
}

any_file_exists() {
  local base_dir="$1"
  shift
  local rel
  for rel in "$@"; do
    if [ -f "${base_dir}/${rel}" ]; then
      return 0
    fi
  done
  return 1
}

sensevoice_dir_ready() {
  local base_dir="$1"
  all_required_files_exist \
    "${base_dir}" \
    "am.mvn" \
    "config.yaml" \
    "configuration.json" \
    "sensevoice_decoder_model.onnx" \
    "tokenizer.vocab" \
    "tokens.txt"

  local meta_ready=$?
  if [ "${meta_ready}" -ne 0 ]; then
    return 1
  fi

  any_file_exists \
    "${base_dir}" \
    "model_quant.onnx" \
    "model_quant_optimized.onnx"
}

melotts_dir_ready() {
  local base_dir="$1"
  all_required_files_exist \
    "${base_dir}" \
    "encoder-zh.onnx" \
    "decoder-zh.onnx" \
    "g-en.bin" \
    "g-jp.bin" \
    "g-zh_mix_en.bin" \
    "lexicon.txt" \
    "tokens.txt"
}

matcha_dir_ready() {
  local base_dir="$1"
  all_required_files_exist \
    "${base_dir}" \
    "matcha-icefall-zh-baker/lexicon.txt" \
    "matcha-icefall-zh-baker/model-steps-3.q.onnx" \
    "matcha-icefall-zh-baker/tokens.txt" \
    "vocos-22khz-univ.q.onnx"
}

copy_seed_if_missing() {
  local label="$1"
  local seed_dir="$2"
  local target_dir="$3"
  local validator="$4"

  if [ -d "${target_dir}" ] && "${validator}" "${target_dir}"; then
    echo "[OK] ${label} already ready at ${target_dir}"
    return 0
  fi

  if [ ! -d "${seed_dir}" ]; then
    echo "[MISS] ${label} seed dir not found: ${seed_dir}"
    return 1
  fi

  if ! "${validator}" "${seed_dir}"; then
    echo "[MISS] ${label} seed dir exists but required local weight files are incomplete: ${seed_dir}"
    echo "       请先把 ${label} 的本地 bootstrap 权重补到该目录，或直接把正式模型同步到 /mnt/ssd/models"
    return 1
  fi

  mkdir -p "${target_dir}"
  cp -a "${seed_dir}/." "${target_dir}/"
  echo "[OK] bootstrapped ${label} from ${seed_dir} -> ${target_dir}"
}

ensure_optional_dir() {
  local label="$1"
  local seed_dir="$2"
  local target_dir="$3"
  local validator="$4"

  if copy_seed_if_missing "${label}" "${seed_dir}" "${target_dir}" "${validator}"; then
    return 0
  fi

  echo "[SKIP] ${label} is optional; keep current state."
  return 0
}

sync_official_optimized_asr_model() {
  local target_dir="$1"
  local official_model="/usr/share/spacemit-asr/sensevoice/model_quant_optimized.onnx"

  if [ ! -f "${official_model}" ]; then
    echo "[SKIP] official optimized ASR model not found at ${official_model}"
    return 0
  fi

  mkdir -p "${target_dir}"
  cp -f "${official_model}" "${target_dir}/model_quant_optimized.onnx"
  echo "[OK] synced official optimized ASR model -> ${target_dir}/model_quant_optimized.onnx"
}

mkdir -p /mnt/ssd/models/asr /mnt/ssd/models/tts /mnt/ssd/models/legacy

copy_seed_if_missing \
  "sensevoice-small" \
  "${PROJECT_ROOT}/assets/bootstrap/asr/sensevoice-small" \
  "/mnt/ssd/models/asr/sensevoice-small" \
  sensevoice_dir_ready

sync_official_optimized_asr_model "/mnt/ssd/models/asr/sensevoice-small"

ensure_optional_dir \
  "melotts" \
  "${PROJECT_ROOT}/assets/bootstrap/tts/melotts" \
  "/mnt/ssd/models/legacy/melotts" \
  melotts_dir_ready

copy_seed_if_missing \
  "matcha-tts" \
  "${HOME}/.cache/models/tts/matcha-tts" \
  "/mnt/ssd/models/tts/matcha-tts" \
  matcha_dir_ready

echo "[DONE] model directory check finished"
