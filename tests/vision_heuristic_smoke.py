from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import VisionConfig, VisionMipiConfig, VisionRecognizerConfig, VisionUsbConfig
from app.vision.recognizer import HeuristicDefectRecognizer


def main():
    with tempfile.TemporaryDirectory(prefix="spacemit-vision-smoke-") as tmp_dir:
        root = Path(tmp_dir)
        image_path = root / "synthetic_rust.png"
        annotated_path = root / "annotated.png"

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :] = [70, 80, 90]
        image[40:180, 60:260] = [168, 92, 44]
        Image.fromarray(image, mode="RGB").save(image_path)

        config = VisionConfig(
            enabled=True,
            backend="mipi_official",
            output_dir=root,
            keep_capture_artifacts=False,
            mipi=VisionMipiConfig(
                detect_json_candidates=[],
                auto_json_dir=root,
                capture_tmp_dir=root,
                capture_timeout_seconds=1,
                prefer_sensor="",
            ),
            usb=VisionUsbConfig(
                device="/dev/video0",
                width=640,
                height=480,
                pixel_format="MJPG",
                capture_timeout_seconds=1,
            ),
            recognizer=VisionRecognizerConfig(
                backend="heuristic_defect",
                spacemit_vision_config=None,
                spacemit_model_path=None,
                lazy_load=False,
                save_annotated_image=True,
                max_candidates=3,
            ),
        )

        recognizer = HeuristicDefectRecognizer(config)
        result = recognizer.analyze(image_path, annotated_path)
        print(result.to_dict())


if __name__ == "__main__":
    main()
