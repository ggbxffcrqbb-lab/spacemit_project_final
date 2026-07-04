from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.semantic_mapping import map_official_detections_to_project_candidates
from app.vision.types import DefectCandidate


def main():
    rust_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rust_rgb[:, :] = [70, 80, 90]
    rust_rgb[40:180, 60:260] = [168, 92, 44]

    fallback_candidates, fallback_metrics, fallback_notes = map_official_detections_to_project_candidates(
        raw_candidates=[
            DefectCandidate(
                label="person",
                score=0.91,
                summary="raw",
                box={"x1": 10.0, "y1": 10.0, "x2": 100.0, "y2": 200.0},
            ),
            DefectCandidate(
                label="kite",
                score=0.68,
                summary="raw",
                box={"x1": 50.0, "y1": 30.0, "x2": 220.0, "y2": 190.0},
            ),
        ],
        rgb=rust_rgb,
        max_candidates=3,
    )
    assert fallback_metrics["mapping_strategy"] == "heuristic_fallback"
    assert fallback_candidates
    assert fallback_candidates[0].label == "rust_like_corrosion"

    direct_candidates, direct_metrics, direct_notes = map_official_detections_to_project_candidates(
        raw_candidates=[
            DefectCandidate(
                label="rust",
                score=0.88,
                summary="raw",
                box={"x1": 12.0, "y1": 22.0, "x2": 90.0, "y2": 140.0},
            ),
            DefectCandidate(
                label="corrosion",
                score=0.67,
                summary="raw",
                box={"x1": 20.0, "y1": 40.0, "x2": 88.0, "y2": 150.0},
            ),
        ],
        rgb=rust_rgb,
        max_candidates=3,
    )
    assert direct_metrics["mapping_strategy"] == "direct_label_map"
    assert direct_candidates
    assert direct_candidates[0].label == "rust_like_corrosion"

    print(
        {
            "fallback": {
                "metrics": fallback_metrics,
                "notes": fallback_notes,
                "candidates": [candidate.to_dict() for candidate in fallback_candidates],
            },
            "direct": {
                "metrics": direct_metrics,
                "notes": direct_notes,
                "candidates": [candidate.to_dict() for candidate in direct_candidates],
            },
        }
    )


if __name__ == "__main__":
    main()
