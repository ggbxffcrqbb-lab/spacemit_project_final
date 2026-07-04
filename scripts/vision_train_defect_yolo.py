from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a defect-specific YOLOv8 model and optionally export ONNX.",
    )
    parser.add_argument(
        "--data",
        default="data/vision/defect_dataset/yolo_v1/data.yaml",
        help="Ultralytics data.yaml path",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base Ultralytics checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device or cpu")
    parser.add_argument("--project", default="runs/defect_train")
    parser.add_argument("--name", default="yolov8n_defect_v1")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export best.pt to ONNX after training",
    )
    parser.add_argument(
        "--emit-xquant-config",
        action="store_true",
        help="Emit an xquant config template for the exported ONNX model",
    )
    parser.add_argument(
        "--xquant-config-path",
        default="",
        help="Optional output path for the generated xquant config json",
    )
    parser.add_argument(
        "--calib-list-path",
        default="data/vision/defect_dataset/calibration/calib_list.txt",
        help="Calibration list path written into the xquant config template",
    )
    return parser.parse_args()


def build_xquant_config(onnx_path: Path, calib_list_path: str) -> dict:
    quantized_name = onnx_path.name.replace(".onnx", ".q.onnx")
    return {
        "model_parameters": {
            "onnx_model": quantized_name,
            "working_dir": "./tmp",
            "skip_onnxsim": False,
        },
        "calibration_parameters": {
            "calibration_type": "minmax",
            "input_parametres": [
                {
                    "mean_value": [0, 0, 0],
                    "std_value": [255, 255, 255],
                    "color_format": "rgb",
                    "data_list_path": calib_list_path,
                }
            ],
        },
        "quantization_parameters": {
            "finetune_level": 3,
            "max_percentile": 0.9995,
            "truncate_var_names": [
                "/model.22/Reshape_output_0",
                "/model.22/Reshape_1_output_0",
                "/model.22/Reshape_2_output_0",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ultralytics。请在训练环境执行 `pip install ultralytics` 后重试。"
        ) from exc

    data_path = Path(args.data).expanduser().resolve()
    model = YOLO(args.model)
    train_result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_pt = Path(train_result.save_dir) / "weights" / "best.pt"
    output = {
        "save_dir": str(train_result.save_dir),
        "best_pt": str(best_pt),
    }

    if args.export_onnx:
        exported = YOLO(str(best_pt)).export(format="onnx")
        onnx_path = Path(str(exported)).expanduser().resolve()
        output["onnx_path"] = str(onnx_path)
        if args.emit_xquant_config:
            xquant_path = (
                Path(args.xquant_config_path).expanduser().resolve()
                if args.xquant_config_path
                else onnx_path.with_name(f"{onnx_path.stem}_xquant_config.json")
            )
            xquant_payload = build_xquant_config(onnx_path, args.calib_list_path)
            xquant_path.write_text(
                json.dumps(xquant_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            output["xquant_config_path"] = str(xquant_path)
            output["expected_quantized_model_name"] = onnx_path.name.replace(".onnx", ".q.onnx")

    print(output)


if __name__ == "__main__":
    main()
