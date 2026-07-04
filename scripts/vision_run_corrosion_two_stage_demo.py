from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import spacemit_ort as _  # noqa: F401  register SpaceMIT EP before onnxruntime
except ImportError:
    pass

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


DEFAULT_SEG_CONFIG = Path("/mnt/ssd/spacemit_project/configs/vision_spacemit_corrosion_seg_1cls_v1_cpu_float.yaml")
DEFAULT_IMAGE_PATH = Path("/mnt/ssd/data/vision/corrosion_two_stage/samples/seg_val_sample.jpg")
DEFAULT_CLS_MODEL = Path("/mnt/ssd/models/vision/corrosion_two_stage/cls/yolov8n_cls_corrosion_3cls_v1.float.onnx")
DEFAULT_CLS_LABELS = Path("/mnt/ssd/spacemit_project/assets/labels/corrosion_cls_3cls.txt")
DEFAULT_OUTPUT_ROOT = Path("/mnt/ssd/data/vision/corrosion_two_stage/demo_outputs")
DEFAULT_SEG_LABEL = "corrosion"
PALETTE = [
    (54, 92, 255),
    (0, 168, 255),
    (0, 196, 154),
    (153, 102, 255),
    (0, 214, 255),
    (255, 160, 0),
]


@dataclass
class ClassificationResult:
    label: str
    score: float
    probabilities: list[float]
    infer_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run board-side corrosion two-stage demo: 1-class segmentation + 3-class classification.",
    )
    parser.add_argument("--seg-config", default=str(DEFAULT_SEG_CONFIG))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--cls-model", default=str(DEFAULT_CLS_MODEL))
    parser.add_argument("--cls-labels", default=str(DEFAULT_CLS_LABELS))
    parser.add_argument("--cls-provider", default="CPUExecutionProvider")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--padding-ratio", type=float, default=0.15)
    parser.add_argument("--min-crop-size", type=int, default=12)
    parser.add_argument("--seg-score-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=32)
    return parser.parse_args()


def load_labels(label_path: Path) -> list[str]:
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise RuntimeError(f"empty label file: {label_path}")
    return labels


def sanitize_name(text: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    return normalized.strip("._") or "item"


def build_run_dir(output_root: Path, run_name: str, image_path: Path) -> Path:
    if run_name:
        resolved_name = sanitize_name(run_name)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_name = f"{image_path.stem}_{stamp}"
    run_dir = output_root / resolved_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_cls_session(model_path: Path, provider: str) -> tuple[ort.InferenceSession, str, str]:
    available = ort.get_available_providers()
    chosen_provider = provider if provider in available else "CPUExecutionProvider"
    session = ort.InferenceSession(str(model_path), providers=[chosen_provider])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def resize_shortest_edge(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {(width, height)}")

    if width < height:
        new_width = size
        new_height = int(round(height * size / width))
    else:
        new_height = size
        new_width = int(round(width * size / height))
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, int(round((width - size) / 2.0)))
    top = max(0, int(round((height - size) / 2.0)))
    right = min(width, left + size)
    bottom = min(height, top + size)
    cropped = image.crop((left, top, right, bottom))
    if cropped.size != (size, size):
        canvas = Image.new("RGB", (size, size))
        paste_x = max(0, (size - cropped.size[0]) // 2)
        paste_y = max(0, (size - cropped.size[1]) // 2)
        canvas.paste(cropped, (paste_x, paste_y))
        return canvas
    return cropped


def preprocess_cls_image(rgb_crop: np.ndarray, size: int = 224) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb_crop, dtype=np.uint8), mode="RGB")
    image = resize_shortest_edge(image, size)
    image = center_crop(image, size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(array, dtype=np.float32)


def softmax(array: np.ndarray) -> np.ndarray:
    shifted = array - np.max(array)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def ensure_probabilities(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    looks_like_probs = (
        np.all(output >= -1e-6)
        and np.all(output <= 1.0 + 1e-6)
        and abs(float(output.sum()) - 1.0) <= 1e-3
    )
    return output if looks_like_probs else softmax(output)


def classify_crop(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    class_names: list[str],
    rgb_crop: np.ndarray,
) -> ClassificationResult:
    tensor = preprocess_cls_image(rgb_crop, size=224)
    start_time = time.perf_counter()
    raw = session.run([output_name], {input_name: tensor})[0][0]
    infer_ms = (time.perf_counter() - start_time) * 1000.0
    probs = ensure_probabilities(raw)
    top1 = int(np.argmax(probs))
    return ClassificationResult(
        label=class_names[top1],
        score=float(probs[top1]),
        probabilities=[float(value) for value in probs.tolist()],
        infer_ms=round(infer_ms, 3),
    )


def build_crop_box_from_mask(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    fallback_box: tuple[float, float, float, float],
    padding_ratio: float,
    min_crop_size: int,
) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size > 0 and ys.size > 0:
        min_x = float(xs.min())
        max_x = float(xs.max())
        min_y = float(ys.min())
        max_y = float(ys.max())
    else:
        min_x, min_y, max_x, max_y = fallback_box

    box_width = max(1.0, max_x - min_x + 1.0)
    box_height = max(1.0, max_y - min_y + 1.0)
    pad_x = max(int(round(box_width * padding_ratio)), max(1, min_crop_size // 4))
    pad_y = max(int(round(box_height * padding_ratio)), max(1, min_crop_size // 4))

    left = max(0, int(np.floor(min_x)) - pad_x)
    top = max(0, int(np.floor(min_y)) - pad_y)
    right = min(width, int(np.ceil(max_x)) + pad_x + 1)
    bottom = min(height, int(np.ceil(max_y)) + pad_y + 1)

    if right - left < min_crop_size:
        deficit = min_crop_size - (right - left)
        left = max(0, left - deficit // 2)
        right = min(width, right + deficit - deficit // 2)
    if bottom - top < min_crop_size:
        deficit = min_crop_size - (bottom - top)
        top = max(0, top - deficit // 2)
        bottom = min(height, bottom + deficit - deficit // 2)

    return left, top, right, bottom


def overlay_binary_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> None:
    active = mask > 0
    if not np.any(active):
        return
    overlay = np.zeros_like(image_bgr, dtype=np.uint8)
    overlay[active] = color
    blended = cv2.addWeighted(image_bgr, 1.0, overlay, alpha, 0.0)
    image_bgr[active] = blended[active]


def draw_label(image_bgr: np.ndarray, origin: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    image_h, image_w = image_bgr.shape[:2]
    x = max(0, min(x, max(0, image_w - text_w - 8)))
    y = max(text_h + 6, min(y, max(text_h + 6, image_h - baseline - 2)))
    top_left = (x, y - text_h - 6)
    bottom_right = (x + text_w + 8, y + baseline - 2)
    cv2.rectangle(image_bgr, top_left, bottom_right, color, thickness=-1)
    cv2.putText(image_bgr, text, (x + 4, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def save_image(path: Path, bgr_image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr_image):
        raise RuntimeError(f"failed to save image: {path}")
    return path


def save_mask(path: Path, mask: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"failed to save mask: {path}")
    return path


def main() -> None:
    args = parse_args()
    seg_config = Path(args.seg_config).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    cls_model = Path(args.cls_model).expanduser().resolve()
    cls_labels = Path(args.cls_labels).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not seg_config.exists():
        raise FileNotFoundError(f"seg config not found: {seg_config}")
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not cls_model.exists():
        raise FileNotFoundError(f"cls model not found: {cls_model}")
    if not cls_labels.exists():
        raise FileNotFoundError(f"cls labels not found: {cls_labels}")

    run_dir = build_run_dir(output_root, args.run_name, image_path)
    crops_dir = run_dir / "crops"
    masks_dir = run_dir / "masks"

    class_names = load_labels(cls_labels)

    from spacemit_vision import VisionServiceNative, VisionServiceStatus

    # Keep the board-side segmentation path aligned with the official example.
    # The simpler factory call is compatible with both float and q.onnx models.
    seg_service = VisionServiceNative.create(str(seg_config), model_path_override="")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    stage1_start = time.perf_counter()
    status, results = seg_service.infer_image(
        bgr,
        conf=args.seg_score_threshold if args.seg_score_threshold is not None else -1.0,
        iou=-1.0,
    )
    stage1_ms = (time.perf_counter() - stage1_start) * 1000.0
    status_name = getattr(status, "name", str(status))
    if status != VisionServiceStatus.OK:
        error_text = ""
        if hasattr(seg_service, "last_error"):
            try:
                error_text = str(seg_service.last_error()).strip()
            except Exception:
                error_text = ""
        detail = f", error={error_text}" if error_text else ""
        raise RuntimeError(f"segmentation failed with status={status_name}{detail}")

    # Release segmentation-side AI core resources before creating the
    # classification session with SpaceMIT EP.
    del seg_service

    cls_session, cls_input_name, cls_output_name = make_cls_session(cls_model, args.cls_provider)

    sorted_results = sorted(results, key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)
    stage1_vis = bgr.copy()
    stage2_vis = bgr.copy()

    detections: list[dict[str, object]] = []
    cls_total_ms = 0.0

    for index, item in enumerate(sorted_results[: max(1, args.max_detections)]):
        seg_score = float(getattr(item, "score", 0.0))
        if seg_score < args.seg_score_threshold:
            continue

        x1 = float(getattr(item, "x1", 0.0))
        y1 = float(getattr(item, "y1", 0.0))
        x2 = float(getattr(item, "x2", 0.0))
        y2 = float(getattr(item, "y2", 0.0))
        bbox_xyxy = (
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(0, min(width - 1, int(round(x2)))),
            max(0, min(height - 1, int(round(y2)))),
        )

        raw_mask = np.asarray(getattr(item, "mask", None), dtype=np.uint8)
        if raw_mask.ndim != 2:
            raw_mask = np.zeros((height, width), dtype=np.uint8)
        if raw_mask.shape != (height, width):
            resized_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            raw_mask = (resized_mask > 0).astype(np.uint8) * 255

        crop_left, crop_top, crop_right, crop_bottom = build_crop_box_from_mask(
            raw_mask,
            width=width,
            height=height,
            fallback_box=(x1, y1, x2, y2),
            padding_ratio=args.padding_ratio,
            min_crop_size=args.min_crop_size,
        )

        crop_rgb = rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
        crop_mask = raw_mask[crop_top:crop_bottom, crop_left:crop_right].copy()

        cls_result = classify_crop(
            cls_session,
            cls_input_name,
            cls_output_name,
            class_names,
            crop_rgb,
        )
        cls_total_ms += cls_result.infer_ms

        color = PALETTE[index % len(PALETTE)]
        overlay_binary_mask(stage1_vis, raw_mask, color)
        overlay_binary_mask(stage2_vis, raw_mask, color)
        cv2.rectangle(stage1_vis, bbox_xyxy[:2], bbox_xyxy[2:], color, 2)
        cv2.rectangle(stage2_vis, bbox_xyxy[:2], bbox_xyxy[2:], color, 2)

        seg_text = f"{DEFAULT_SEG_LABEL} {seg_score:.2f}"
        cls_text = f"{cls_result.label} {cls_result.score:.2f}"
        draw_label(stage1_vis, (bbox_xyxy[0], bbox_xyxy[1] - 6), seg_text, color)
        draw_label(stage2_vis, (bbox_xyxy[0], bbox_xyxy[1] - 6), seg_text, color)
        draw_label(stage2_vis, (bbox_xyxy[0], bbox_xyxy[1] + 18), cls_text, color)

        crop_name = f"{index:02d}_{sanitize_name(cls_result.label)}"
        crop_path = save_image(crops_dir / f"{crop_name}.jpg", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        mask_path = save_mask(masks_dir / f"{crop_name}.png", crop_mask)

        detections.append(
            {
                "index": index,
                "segmentation_label": DEFAULT_SEG_LABEL,
                "segmentation_score": round(seg_score, 6),
                "bbox_xyxy": [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]],
                "crop_box_xyxy": [crop_left, crop_top, crop_right, crop_bottom],
                "mask_area_pixels": int(np.count_nonzero(raw_mask)),
                "classification_label": cls_result.label,
                "classification_score": round(cls_result.score, 6),
                "classification_probabilities": {
                    class_name: round(prob, 6)
                    for class_name, prob in zip(class_names, cls_result.probabilities)
                },
                "classification_infer_ms": cls_result.infer_ms,
                "crop_path": str(crop_path),
                "mask_path": str(mask_path),
            }
        )

    stage1_path = save_image(run_dir / "stage1_segmentation.jpg", stage1_vis)
    stage2_path = save_image(run_dir / "stage2_two_stage.jpg", stage2_vis)
    manifest = {
        "run_dir": str(run_dir),
        "image_path": str(image_path),
        "seg_config": str(seg_config),
        "seg_status": status_name,
        "seg_raw_detection_count": len(results),
        "seg_kept_detection_count": len(detections),
        "cls_model": str(cls_model),
        "cls_labels": class_names,
        "cls_provider_requested": args.cls_provider,
        "cls_provider_actual": cls_session.get_providers()[0] if cls_session.get_providers() else "unknown",
        "timing_ms": {
            "stage1_segmentation": round(stage1_ms, 3),
            "stage2_classification_total": round(cls_total_ms, 3),
            "end_to_end_estimated": round(stage1_ms + cls_total_ms, 3),
        },
        "outputs": {
            "stage1_image": str(stage1_path),
            "stage2_image": str(stage2_path),
        },
        "detections": detections,
    }
    manifest_path = run_dir / "result.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "detections": len(detections),
        "stage1_image": str(stage1_path),
        "stage2_image": str(stage2_path),
        "result_json": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
