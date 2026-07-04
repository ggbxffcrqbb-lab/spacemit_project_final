from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import url2pathname

import yaml
from PIL import Image


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_defect_taxonomy(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def get_bbox_class_names(taxonomy: dict[str, Any]) -> list[str]:
    classes = taxonomy.get("bbox_classes", [])
    return [str(item["name"]) for item in classes]


def get_image_level_tags(taxonomy: dict[str, Any]) -> set[str]:
    return {str(item["name"]) for item in taxonomy.get("image_level_tags", [])}


def extract_analysis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "analysis" in payload and isinstance(payload["analysis"], dict):
        return payload["analysis"]
    if "recognizer_backend" in payload:
        return payload
    raise ValueError("未找到可识别的视觉分析结果结构")


def build_label_studio_task(
    payload: dict[str, Any],
    result_json_path: str | Path,
    class_names: list[str],
    *,
    path_mode: str = "absolute",
    document_root: str | Path | None = None,
    from_name: str = "bbox",
    to_name: str = "image",
    model_version: str = "spacemit_vision_bootstrap",
) -> dict[str, Any]:
    analysis = extract_analysis_payload(payload)
    image_path = Path(str(analysis["image_path"])).expanduser()
    width, height = read_image_size(image_path)
    raw_candidates = list(analysis.get("raw_candidates", []))
    mapped_candidates = list(analysis.get("candidates", []))

    prediction_results = []
    for index, candidate in enumerate(raw_candidates):
        label = str(candidate.get("label", ""))
        box = candidate.get("box")
        if label not in class_names or not isinstance(box, dict):
            continue
        prediction_results.append(
            build_rectangle_prediction(
                result_id=f"pred_{index}",
                label=label,
                box=box,
                image_width=width,
                image_height=height,
                from_name=from_name,
                to_name=to_name,
            )
        )

    task = {
        "data": {
            "image": to_label_studio_image_ref(
                image_path,
                path_mode=path_mode,
                document_root=document_root,
            ),
            "source_result_json": str(Path(result_json_path).expanduser()),
            "raw_detection_labels": [candidate.get("label", "") for candidate in raw_candidates],
            "mapped_candidate_labels": [candidate.get("label", "") for candidate in mapped_candidates],
            "notes_text": "\n".join(str(note) for note in analysis.get("notes", [])),
        },
    }
    if prediction_results:
        task["predictions"] = [
            {
                "model_version": model_version,
                "score": max(
                    float(candidate.get("score", 0.0))
                    for candidate in raw_candidates
                    if str(candidate.get("label", "")) in class_names
                ),
                "result": prediction_results,
            }
        ]
    return task


def build_rectangle_prediction(
    *,
    result_id: str,
    label: str,
    box: dict[str, Any],
    image_width: int,
    image_height: int,
    from_name: str,
    to_name: str,
) -> dict[str, Any]:
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))
    left = max(0.0, min(x1, x2))
    top = max(0.0, min(y1, y2))
    width = max(0.0, abs(x2 - x1))
    height = max(0.0, abs(y2 - y1))
    return {
        "id": result_id,
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
        "value": {
            "x": left / image_width * 100.0 if image_width else 0.0,
            "y": top / image_height * 100.0 if image_height else 0.0,
            "width": width / image_width * 100.0 if image_width else 0.0,
            "height": height / image_height * 100.0 if image_height else 0.0,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def to_label_studio_image_ref(
    image_path: str | Path,
    *,
    path_mode: str,
    document_root: str | Path | None = None,
) -> str:
    image_path = Path(image_path).expanduser().resolve()
    if path_mode == "absolute":
        return str(image_path)
    if path_mode == "file-uri":
        return image_path.as_uri()
    if path_mode == "label-studio-local":
        if document_root is None:
            raise ValueError("path_mode=label-studio-local 时必须提供 document_root")
        root = Path(document_root).expanduser().resolve()
        relative = image_path.relative_to(root).as_posix()
        return f"/data/local-files/?d={relative}"
    raise ValueError(f"不支持的 path_mode: {path_mode}")


def read_image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_label_studio_export(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "tasks" in payload and isinstance(payload["tasks"], list):
        return payload["tasks"]
    raise ValueError("Label Studio 导出文件格式不受支持")


def convert_label_studio_export_to_yolo(
    export_path: str | Path,
    dataset_root: str | Path,
    class_names: list[str],
    *,
    document_root: str | Path | None = None,
    from_name: str = "bbox",
    to_name: str = "image",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    tasks = load_label_studio_export(export_path)
    samples: list[dict[str, Any]] = []
    image_tags = set()

    for task_index, task in enumerate(tasks):
        annotation = select_annotation(task)
        if annotation is None:
            continue
        image_path = resolve_task_image_path(task, document_root=document_root)
        if image_path is None or not image_path.exists():
            continue

        yolo_lines: list[str] = []
        for result in annotation.get("result", []):
            if not is_matching_rectangle_result(result, from_name=from_name, to_name=to_name):
                tag_labels = extract_image_level_labels(result)
                image_tags.update(tag_labels)
                continue
            yolo_line = rectangle_result_to_yolo_line(result, class_names)
            if yolo_line:
                yolo_lines.append(yolo_line)

        if not yolo_lines:
            continue

        samples.append(
            {
                "task_index": task_index,
                "image_path": image_path,
                "label_lines": yolo_lines,
                "annotation_id": annotation.get("id", ""),
            }
        )

    if not samples:
        raise ValueError("未从 Label Studio 导出中解析出任何可用的矩形标注")

    dataset_root = Path(dataset_root).expanduser().resolve()
    splits = split_samples(samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    manifest_rows: list[dict[str, Any]] = []
    total_written = 0

    for split_name, split_samples_list in splits.items():
        for sample_index, sample in enumerate(split_samples_list):
            image_src = Path(sample["image_path"])
            target_stem = f"{sample['task_index']:05d}_{sample_index:03d}_{image_src.stem}"
            image_dst = dataset_root / "images" / split_name / f"{target_stem}{image_src.suffix.lower()}"
            label_dst = dataset_root / "labels" / split_name / f"{target_stem}.txt"
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_src, image_dst)
            label_dst.write_text("\n".join(sample["label_lines"]) + "\n", encoding="utf-8")
            manifest_rows.append(
                {
                    "split": split_name,
                    "image_src": str(image_src),
                    "image_dst": str(image_dst),
                    "label_dst": str(label_dst),
                    "annotation_id": sample["annotation_id"],
                }
            )
            total_written += 1

    write_ultralytics_data_yaml(dataset_root, class_names)
    save_json(dataset_root / "manifest.json", manifest_rows)
    (dataset_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    return {
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "image_level_tags_seen": sorted(image_tags),
        "num_samples": total_written,
        "splits": {name: len(items) for name, items in splits.items()},
        "manifest_path": str(dataset_root / "manifest.json"),
        "data_yaml_path": str(dataset_root / "data.yaml"),
    }


def select_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = task.get("annotations", [])
    if isinstance(annotations, list):
        for annotation in reversed(annotations):
            if annotation.get("result"):
                return annotation
    return None


def resolve_task_image_path(
    task: dict[str, Any],
    *,
    document_root: str | Path | None = None,
) -> Path | None:
    data = task.get("data", {})
    raw_value = data.get("image")
    if not raw_value:
        return None
    raw = str(raw_value)
    if raw.startswith("file://"):
        return Path(url2pathname(urlparse(raw).path))
    if raw.startswith("/data/local-files/"):
        if document_root is None:
            raise ValueError("导出中使用了 /data/local-files 路径，必须提供 document_root")
        query = parse_qs(urlparse(raw).query)
        relative = unquote(query.get("d", [""])[0])
        return Path(document_root).expanduser().resolve() / relative
    return Path(raw).expanduser()


def is_matching_rectangle_result(
    result: dict[str, Any],
    *,
    from_name: str,
    to_name: str,
) -> bool:
    return (
        result.get("type") == "rectanglelabels"
        and result.get("from_name") == from_name
        and result.get("to_name") == to_name
        and isinstance(result.get("value"), dict)
        and bool(result["value"].get("rectanglelabels"))
    )


def extract_image_level_labels(result: dict[str, Any]) -> list[str]:
    if result.get("type") != "choices":
        return []
    value = result.get("value", {})
    choices = value.get("choices", [])
    if not isinstance(choices, list):
        return []
    return [str(choice) for choice in choices]


def rectangle_result_to_yolo_line(result: dict[str, Any], class_names: list[str]) -> str | None:
    value = result["value"]
    labels = value.get("rectanglelabels", [])
    if not labels:
        return None
    label = str(labels[0])
    if label not in class_names:
        return None
    class_id = class_names.index(label)
    x = float(value.get("x", 0.0)) / 100.0
    y = float(value.get("y", 0.0)) / 100.0
    width = float(value.get("width", 0.0)) / 100.0
    height = float(value.get("height", 0.0)) / 100.0
    x_center = x + width / 2.0
    y_center = y + height / 2.0
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def split_samples(
    samples: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio/val_ratio 配置非法")

    samples_copy = list(samples)
    random.Random(seed).shuffle(samples_copy)
    total = len(samples_copy)
    train_count = max(1, int(total * train_ratio))
    val_count = int(total * val_ratio)
    if total >= 2 and val_count == 0:
        val_count = 1
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    test_count = total - train_count - val_count
    if total >= 3 and test_count == 0 and train_count > 1:
        train_count -= 1
        test_count = 1

    train_split = samples_copy[:train_count]
    val_split = samples_copy[train_count: train_count + val_count]
    test_split = samples_copy[train_count + val_count:]

    return {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }


def write_ultralytics_data_yaml(dataset_root: str | Path, class_names: list[str]) -> None:
    dataset_root = Path(dataset_root)
    payload = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (dataset_root / "data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
