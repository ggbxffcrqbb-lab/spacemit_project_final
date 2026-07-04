from __future__ import annotations

import tempfile
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.dataset_tools import (
    build_label_studio_task,
    convert_label_studio_export_to_yolo,
    get_bbox_class_names,
    load_defect_taxonomy,
    save_json,
)


def main() -> None:
    taxonomy = load_defect_taxonomy(PROJECT_ROOT / "configs" / "defect_taxonomy.yaml")
    class_names = get_bbox_class_names(taxonomy)

    with tempfile.TemporaryDirectory(prefix="spacemit-dataset-tools-") as tmp_dir:
        root = Path(tmp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (320, 240), color=(160, 120, 80)).save(image_path)

        payload = {
            "recognizer_backend": "spacemit_vision",
            "image_path": str(image_path),
            "notes": ["test"],
            "candidates": [],
            "raw_candidates": [
                {
                    "label": "rust_like_corrosion",
                    "score": 0.91,
                    "box": {"x1": 32.0, "y1": 48.0, "x2": 160.0, "y2": 144.0},
                    "summary": "raw",
                    "evidence": {},
                }
            ],
        }
        result_json = root / "result.json"
        save_json(result_json, payload)

        task = build_label_studio_task(
            payload,
            result_json_path=result_json,
            class_names=class_names,
            path_mode="absolute",
        )
        assert task["predictions"]

        export_json = root / "labelstudio_export.json"
        save_json(
            export_json,
            [
                {
                    "data": task["data"],
                    "annotations": [
                        {
                            "id": 1,
                            "result": [
                                {
                                    "id": "box-1",
                                    "from_name": "bbox",
                                    "to_name": "image",
                                    "type": "rectanglelabels",
                                    "original_width": 320,
                                    "original_height": 240,
                                    "image_rotation": 0,
                                    "value": {
                                        "x": 10.0,
                                        "y": 20.0,
                                        "width": 30.0,
                                        "height": 25.0,
                                        "rotation": 0,
                                        "rectanglelabels": ["rust_like_corrosion"],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        )

        dataset_root = root / "yolo_dataset"
        result = convert_label_studio_export_to_yolo(
            export_path=export_json,
            dataset_root=dataset_root,
            class_names=class_names,
        )
        assert result["num_samples"] == 1
        assert (dataset_root / "data.yaml").exists()
        assert list((dataset_root / "labels").rglob("*.txt"))

        print(result)


if __name__ == "__main__":
    main()
