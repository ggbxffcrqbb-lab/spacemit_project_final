from __future__ import annotations

import argparse

from app.vision.dataset_tools import (
    convert_label_studio_export_to_yolo,
    get_bbox_class_names,
    load_defect_taxonomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Label Studio export json to an Ultralytics YOLO dataset.",
    )
    parser.add_argument("export_json", help="Label Studio export json path")
    parser.add_argument(
        "--dataset-root",
        default="data/vision/defect_dataset/yolo_v1",
        help="Output YOLO dataset root",
    )
    parser.add_argument(
        "--taxonomy",
        default="configs/defect_taxonomy.yaml",
        help="Defect taxonomy yaml path",
    )
    parser.add_argument(
        "--document-root",
        default="",
        help="Required when Label Studio tasks use /data/local-files paths",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = load_defect_taxonomy(args.taxonomy)
    class_names = get_bbox_class_names(taxonomy)
    result = convert_label_studio_export_to_yolo(
        export_path=args.export_json,
        dataset_root=args.dataset_root,
        class_names=class_names,
        document_root=args.document_root or None,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
