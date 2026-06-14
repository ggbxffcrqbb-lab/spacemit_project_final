#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mnt/ssd/logs/tts_exhaust/resident
export SPACEMIT_TTS_TRACE=0

python3 /mnt/ssd/spacemit_project/benchmarks/tts_exhaust/bench_tts_resident.py \
  --repo-root /mnt/ssd/spacemit_project/third_party/model-zoo-tts \
  --preset matcha_zh \
  --provider cpu \
  --threads 3 \
  --requests 3 \
  --text-key zh_medium \
  --warmup \
  --jsonl /mnt/ssd/logs/tts_exhaust/resident/medium_resident_candidates_v2.jsonl \
  --tag matcha_zh_t3_warmon_medium >/dev/null

python3 /mnt/ssd/spacemit_project/benchmarks/tts_exhaust/bench_tts_resident.py \
  --repo-root /mnt/ssd/spacemit_project/third_party/model-zoo-tts \
  --preset matcha_zh \
  --provider cpu \
  --threads 4 \
  --requests 3 \
  --text-key zh_medium \
  --warmup \
  --jsonl /mnt/ssd/logs/tts_exhaust/resident/medium_resident_candidates_v2.jsonl \
  --tag matcha_zh_t4_warmon_medium >/dev/null

python3 /mnt/ssd/spacemit_project/benchmarks/tts_exhaust/bench_tts_resident.py \
  --repo-root /mnt/ssd/spacemit_project/third_party/model-zoo-tts \
  --preset matcha_zh_en \
  --provider cpu \
  --threads 3 \
  --requests 3 \
  --text-key zh_medium \
  --warmup \
  --jsonl /mnt/ssd/logs/tts_exhaust/resident/medium_resident_candidates_v2.jsonl \
  --tag matcha_zh_en_t3_warmon_medium >/dev/null

python3 /mnt/ssd/spacemit_project/benchmarks/tts_exhaust/bench_tts_resident.py \
  --repo-root /mnt/ssd/spacemit_project/third_party/model-zoo-tts \
  --preset matcha_zh_en \
  --provider cpu \
  --threads 4 \
  --requests 3 \
  --text-key zh_medium \
  --warmup \
  --jsonl /mnt/ssd/logs/tts_exhaust/resident/medium_resident_candidates_v2.jsonl \
  --tag matcha_zh_en_t4_warmon_medium >/dev/null
