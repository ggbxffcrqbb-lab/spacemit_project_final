from __future__ import annotations

import argparse
from pathlib import Path

from app.vision.dataset_tools import (
    build_label_studio_task,
    get_bbox_class_names,
    load_defect_taxonomy,
    load_json,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Label Studio tasks from saved vision result json files.",
    )
    parser.add_argument("inputs", nargs="+", help="Result json files or directories")
    parser.add_argument(
        "--taxonomy",
        default="configs/defect_taxonomy.yaml",
        help="Defect taxonomy yaml path",
    )
    parser.add_argument(
        "--output",
        default="data/vision/label_studio/tasks.json",
        help="Output tasks json path",
    )
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "file-uri", "label-studio-local"],
        default="absolute",
        help="How to encode image path inside Label Studio tasks",
    )
    parser.add_argument(
        "--document-root",
        default="",
        help="Required when --path-mode=label-studio-local",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = load_defect_taxonomy(args.taxonomy)
    class_names = get_bbox_class_names(taxonomy)

    result_files = sorted(iter_result_json_files(args.inputs))
    tasks = []
    for result_file in result_files:
        payload = load_json(result_file)
        tasks.append(
            build_label_studio_task(
                payload,
                result_json_path=result_file,
                class_names=class_names,
                path_mode=args.path_mode,
                document_root=args.document_root or None,
            )
        )

    save_json(args.output, tasks)
    print(
        {
            "num_tasks": len(tasks),
            "output": str(Path(args.output).expanduser()),
            "taxonomy": str(Path(args.taxonomy).expanduser()),
        }
    )


def iter_result_json_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
    return files


if __name__ == "__main__":
    main()
