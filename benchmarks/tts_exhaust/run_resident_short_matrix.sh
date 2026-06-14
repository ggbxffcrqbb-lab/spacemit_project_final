#!/usr/bin/env bash
set -euo pipefail

mkdir -p /mnt/ssd/logs/tts_exhaust/resident
export SPACEMIT_TTS_TRACE=0

for preset in matcha_zh matcha_zh_en; do
  for threads in 1 2 3 4; do
    for warm in off on; do
      extra=()
      tag_warm="warmoff"
      if [[ "$warm" == "on" ]]; then
        extra+=(--warmup)
        tag_warm="warmon"
      fi

      python3 /mnt/ssd/spacemit_project/benchmarks/tts_exhaust/bench_tts_resident.py \
        --repo-root /mnt/ssd/spacemit_project/third_party/model-zoo-tts \
        --preset "$preset" \
        --provider cpu \
        --threads "$threads" \
        --requests 4 \
        --text-key zh_short \
        "${extra[@]}" \
        --jsonl /mnt/ssd/logs/tts_exhaust/resident/short_resident_matrix.jsonl \
        --tag "${preset}_t${threads}_${tag_warm}" >/dev/null
    done
  done
done
