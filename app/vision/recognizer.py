from __future__ import annotations

import abc
import logging
import os
import queue
import threading
import time
from pathlib import Path

try:
    import spacemit_ort as _  # noqa: F401  register SpaceMIT EP before onnxruntime
except ImportError:
    pass

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from app.core.cpu_affinity import affinity_scope, bind_current_thread
from app.core.config import VisionConfig
from app.vision.image_utils import bgr_to_rgb, load_rgb_image, rgb_to_bgr, save_rgb_image
from app.vision.semantic_mapping import (
    build_heuristic_defect_candidates,
    map_official_detections_to_project_candidates,
)
from app.vision.types import DefectCandidate, VisionAnalysisResult


class DefectRecognizer(abc.ABC):
    def __init__(self, config: VisionConfig):
        self.config = config
        self.logger = logging.getLogger(f"app.vision.recognizer.{self.backend_name}")

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    def analyze(self, image_path: Path, annotated_output_path: Path | None = None) -> VisionAnalysisResult:
        rgb = load_rgb_image(image_path)
        return self.analyze_rgb(rgb, image_path, annotated_output_path)

    @abc.abstractmethod
    def analyze_rgb(
        self,
        rgb: np.ndarray,
        image_path: Path,
        annotated_output_path: Path | None = None,
    ) -> VisionAnalysisResult:
        raise NotImplementedError


class HeuristicDefectRecognizer(DefectRecognizer):
    @property
    def backend_name(self) -> str:
        return "heuristic_defect"

    def analyze_rgb(
        self,
        rgb: np.ndarray,
        image_path: Path,
        annotated_output_path: Path | None = None,
    ) -> VisionAnalysisResult:
        candidates, metrics = build_heuristic_defect_candidates(
            rgb,
            self.config.recognizer.max_candidates,
        )

        annotated_path = None
        if self.config.recognizer.save_annotated_image:
            annotated_path = annotated_output_path or self._default_annotated_path(image_path)
            render_candidate_overlay(
                rgb,
                candidates,
                annotated_path,
                header="Phase 5 Heuristic Defect Hints",
            )

        return VisionAnalysisResult(
            recognizer_backend=self.backend_name,
            image_path=image_path,
            annotated_image_path=annotated_path,
            candidates=candidates,
            raw_candidates=[],
            metrics=metrics,
            notes=[
                "Phase 5 heuristic baseline recognizer.",
                "Replace with spacemit_vision or trained ONNX backend.",
            ],
        )

    def _default_annotated_path(self, image_path: Path) -> Path:
        return self.config.output_dir / "annotated" / f"{image_path.stem}_heuristic.png"


class SpacemitVisionRecognizer(DefectRecognizer):
    def __init__(self, config: VisionConfig):
        super().__init__(config)
        self._service = None

    @property
    def backend_name(self) -> str:
        return "spacemit_vision"

    def analyze_rgb(
        self,
        rgb: np.ndarray,
        image_path: Path,
        annotated_output_path: Path | None = None,
    ) -> VisionAnalysisResult:
        service = self._ensure_service()
        bgr = rgb_to_bgr(rgb)
        status, results = service.infer_image(bgr)
        status_name = getattr(status, "name", str(status))
        if status_name != "OK":
            raise RuntimeError(f"spacemit_vision inference failed: {status_name}")
        class_names = self._get_class_names(service)
        raw_candidates = self._build_raw_candidates(results, class_names)
        candidates, mapping_metrics, mapping_notes = map_official_detections_to_project_candidates(
            raw_candidates, rgb, self.config.recognizer.max_candidates,
        )
        annotated_path = None
        if self.config.recognizer.save_annotated_image:
            annotated_path = annotated_output_path or self._default_annotated_path(image_path)
            render_candidate_overlay(rgb, candidates, annotated_path, header="Phase 5 Vision Result")
        return VisionAnalysisResult(
            recognizer_backend=self.backend_name,
            image_path=image_path,
            annotated_image_path=annotated_path,
            candidates=candidates,
            raw_candidates=raw_candidates[: self._raw_candidate_limit()],
            metrics={"status": status_name, "detections": len(raw_candidates), **mapping_metrics},
            notes=["spacemit_vision backend result.", *mapping_notes],
        )

    def _default_annotated_path(self, image_path: Path) -> Path:
        return self.config.output_dir / "annotated" / f"{image_path.stem}_spacemit.png"

    def _ensure_service(self):
        if self._service is not None:
            return self._service
        config_path = self.config.recognizer.spacemit_vision_config
        if not config_path:
            raise RuntimeError("spacemit_vision_config not configured")
        try:
            from spacemit_vision import VisionServiceNative
        except ImportError as exc:
            raise RuntimeError("spacemit_vision not installed") from exc
        self._service = VisionServiceNative.create(
            str(config_path), str(self.config.recognizer.spacemit_model_path or ""),
            self.config.recognizer.lazy_load, True, False,
        )
        return self._service

    @staticmethod
    def _build_raw_candidates(results, class_names):
        raw = []
        for item in results:
            lid = int(getattr(item, "label", -1))
            name = class_names[lid] if 0 <= lid < len(class_names) else f"class_{lid}"
            raw.append(DefectCandidate(
                label=name, score=float(getattr(item, "score", 0.0)),
                summary="raw model detection",
                box={"x1": float(getattr(item, "x1", 0)), "y1": float(getattr(item, "y1", 0)),
                     "x2": float(getattr(item, "x2", 0)), "y2": float(getattr(item, "y2", 0))},
                evidence={"label_id": lid},
            ))
        return raw

    def _raw_candidate_limit(self):
        return max(self.config.recognizer.max_candidates, 8)

    @staticmethod
    def _get_class_names(service):
        try:
            return list(service.get_class_names())
        except Exception:
            return []


class CorrosionTwoStageRealtimeRecognizer(DefectRecognizer):
    DEFAULT_SEG_LABEL = "corrosion"
    DEFAULT_SEG_CONFIG = Path("/mnt/ssd/spacemit_project/configs/vision_spacemit_corrosion_seg_1cls_v1.yaml")
    DEFAULT_CLS_MODEL = Path(
        "/mnt/ssd/models/vision/corrosion_two_stage/cls/yolov8n_cls_corrosion_3cls_v1.q.onnx"
    )
    DEFAULT_CLS_LABELS = Path("/mnt/ssd/spacemit_project/assets/labels/corrosion_cls_3cls.txt")

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        options = dict(config.recognizer.options or {})
        self._seg_service = None
        self._cls_session = None
        self._cls_input_name = ""
        self._cls_output_name = ""
        self._cls_provider_actual = "uninitialized"
        self._seg_warmed = False
        self._cls_warmed = False
        self._track_lock = threading.Lock()
        self._tracks: dict[int, dict[str, object]] = {}
        self._next_track_id = 1
        self._analysis_index = 0
        self._job_queue: queue.Queue[dict[str, object]] = queue.Queue(
            maxsize=max(1, int(options.get("max_pending_jobs", 1)))
        )
        self._worker_started = False
        self._worker_lock = threading.Lock()
        self._seg_config = Path(
            str(
                config.recognizer.spacemit_vision_config
                or options.get("seg_config_path")
                or self.DEFAULT_SEG_CONFIG
            )
        )
        self._seg_model_override = str(config.recognizer.spacemit_model_path or options.get("seg_model_path") or "")
        self._seg_score_threshold = float(options.get("seg_score_threshold", 0.25))
        self._seg_iou_threshold = float(options.get("seg_iou_threshold", -1.0))
        self._padding_ratio = float(options.get("padding_ratio", 0.15))
        self._min_crop_size = int(options.get("min_crop_size", 16))
        self._max_detections = max(1, int(options.get("max_detections", config.recognizer.max_candidates)))
        self._classify_top_k = max(1, int(options.get("classify_top_k", 2)))
        self._classification_cooldown_seconds = float(
            options.get("classification_cooldown_seconds", 1.2)
        )
        self._classification_refresh_iou = float(options.get("classification_refresh_iou", 0.7))
        self._bootstrap_wait_seconds = float(options.get("bootstrap_wait_seconds", 0.35))
        self._track_ttl_seconds = float(options.get("track_ttl_seconds", 2.0))
        self._track_match_iou = float(options.get("track_match_iou", 0.35))
        self._max_display_tracks = max(1, int(options.get("max_display_tracks", config.recognizer.max_candidates)))
        self._cls_model = Path(str(options.get("cls_model_path") or self.DEFAULT_CLS_MODEL))
        self._cls_labels_path = Path(str(options.get("cls_labels_path") or self.DEFAULT_CLS_LABELS))
        self._cls_provider_requested = str(options.get("cls_provider", "CPUExecutionProvider"))
        self._cls_cpu_threads = max(1, int(options.get("cls_cpu_threads", 8)))
        self._eager_cls_session = bool(options.get("eager_cls_session", True))
        self._vision_cpuset = os.getenv("SPACEMIT_VISION_CPUSET", "").strip()
        self._class_names = self._load_labels(self._cls_labels_path)
        if self._eager_cls_session:
            self._ensure_cls_session()
            self._warmup_cls_session()

    @property
    def backend_name(self) -> str:
        return "corrosion_two_stage_rt"

    def analyze_rgb(
        self,
        rgb: np.ndarray,
        image_path: Path,
        annotated_output_path: Path | None = None,
    ) -> VisionAnalysisResult:
        with affinity_scope(self._vision_cpuset):
            self._analysis_index += 1
            self._ensure_worker()
            seg_service = self._ensure_seg_service()
            bgr = rgb_to_bgr(rgb)

            seg_started_at = time.perf_counter()
            status, results = seg_service.infer_image(
                bgr,
                conf=self._seg_score_threshold if self._seg_score_threshold >= 0.0 else -1.0,
                iou=self._seg_iou_threshold if self._seg_iou_threshold >= 0.0 else -1.0,
            )
            seg_infer_ms = (time.perf_counter() - seg_started_at) * 1000.0
            status_name = getattr(status, "name", str(status))
            if status_name != "OK":
                raise RuntimeError(f"corrosion two-stage segmentation failed: {status_name}")

            detections, raw_candidates = self._build_segmentation_detections(
                results,
                rgb,
                image_path,
            )
            bootstrap_event = self._schedule_classification_jobs(detections)
            if bootstrap_event is not None:
                bootstrap_event.wait(timeout=self._bootstrap_wait_seconds)

            candidates = self._build_project_candidates(detections)
            annotated_path = None
            if self.config.recognizer.save_annotated_image:
                annotated_path = annotated_output_path or self._default_annotated_path(image_path)
                self._render_boxes(rgb, candidates, annotated_path)

            metrics = {
                "status": status_name,
                "seg_infer_ms": round(seg_infer_ms, 3),
                "seg_detection_count": len(raw_candidates),
                "display_candidate_count": len(candidates),
                "classify_queue_depth": int(self._job_queue.qsize()),
                "cls_provider_requested": self._cls_provider_requested,
                "cls_provider_actual": self._cls_provider_actual,
                "strategy": "native_live_preview + async_cls_cache",
                "analysis_index": self._analysis_index,
            }
            notes = [
                "Stage-1 corrosion segmentation runs on the board-side SpaceMIT vision service.",
                "Stage-2 subtype classification is cached and refreshed asynchronously for smoother live demo playback.",
            ]
            return VisionAnalysisResult(
                recognizer_backend=self.backend_name,
                image_path=image_path,
                annotated_image_path=annotated_path,
                candidates=candidates,
                raw_candidates=raw_candidates[: max(self._max_display_tracks, 8)],
                metrics=metrics,
                notes=notes,
            )

    def _default_annotated_path(self, image_path: Path) -> Path:
        return self.config.output_dir / "annotated" / f"{image_path.stem}_corrosion_two_stage_rt.png"

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        with self._worker_lock:
            if self._worker_started:
                return
            worker = threading.Thread(
                target=self._classification_worker,
                name="corrosion_two_stage_cls_worker",
                daemon=True,
            )
            worker.start()
            self._worker_started = True

    def _ensure_seg_service(self):
        if self._seg_service is not None:
            return self._seg_service
        if not self._seg_config.exists():
            raise FileNotFoundError(f"segmentation config not found: {self._seg_config}")
        try:
            from spacemit_vision import VisionServiceNative
        except ImportError as exc:
            raise RuntimeError("spacemit_vision not installed") from exc
        self._seg_service = VisionServiceNative.create(
            str(self._seg_config),
            model_path_override=self._seg_model_override,
        )
        self._warmup_seg_service()
        return self._seg_service

    def _ensure_cls_session(self) -> ort.InferenceSession:
        if self._cls_session is not None:
            return self._cls_session
        if not self._cls_model.exists():
            raise FileNotFoundError(f"classification model not found: {self._cls_model}")
        with affinity_scope(self._vision_cpuset, logger=self.logger, label="vision classifier session"):
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self._cls_cpu_threads
            session_options.inter_op_num_threads = 1
            available = set(ort.get_available_providers())
            providers = (
                [self._cls_provider_requested]
                if self._cls_provider_requested in available
                else ["CPUExecutionProvider"]
            )
            self._cls_session = ort.InferenceSession(
                str(self._cls_model),
                sess_options=session_options,
                providers=providers,
            )
            self._cls_provider_actual = (
                self._cls_session.get_providers()[0] if self._cls_session.get_providers() else "unknown"
            )
            self._cls_input_name = self._cls_session.get_inputs()[0].name
            self._cls_output_name = self._cls_session.get_outputs()[0].name
            self.logger.info(
                "corrosion two-stage classifier ready: requested=%s actual=%s threads=%s",
                self._cls_provider_requested,
                self._cls_provider_actual,
                self._cls_cpu_threads,
            )
        return self._cls_session

    def _warmup_seg_service(self) -> None:
        if self._seg_service is None or self._seg_warmed:
            return
        dummy_bgr = np.zeros((320, 320, 3), dtype=np.uint8)
        with affinity_scope(self._vision_cpuset, logger=self.logger, label="vision segmentation warmup"):
            try:
                self._seg_service.infer_image(
                    dummy_bgr,
                    conf=self._seg_score_threshold if self._seg_score_threshold >= 0.0 else -1.0,
                    iou=self._seg_iou_threshold if self._seg_iou_threshold >= 0.0 else -1.0,
                )
                self._seg_warmed = True
                self.logger.info("corrosion two-stage segmentation warmup complete")
            except Exception as exc:
                self.logger.warning("corrosion two-stage segmentation warmup failed: %s", exc)

    def _warmup_cls_session(self) -> None:
        if self._cls_session is None or self._cls_warmed:
            return
        dummy_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
        with affinity_scope(self._vision_cpuset, logger=self.logger, label="vision classifier warmup"):
            try:
                self._cls_session.run(
                    [self._cls_output_name],
                    {self._cls_input_name: dummy_tensor},
                )
                self._cls_warmed = True
                self.logger.info("corrosion two-stage classifier warmup complete")
            except Exception as exc:
                self.logger.warning("corrosion two-stage classifier warmup failed: %s", exc)

    def _build_segmentation_detections(
        self,
        results,
        rgb: np.ndarray,
        image_path: Path,
    ) -> tuple[list[dict[str, object]], list[DefectCandidate]]:
        height, width = rgb.shape[:2]
        ranked = sorted(results, key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)
        now = time.monotonic()
        raw_candidates: list[DefectCandidate] = []
        detections: list[dict[str, object]] = []

        for item in ranked:
            seg_score = float(getattr(item, "score", 0.0))
            if seg_score < self._seg_score_threshold:
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

            crop_box = self._build_crop_box_from_mask(
                raw_mask,
                width=width,
                height=height,
                fallback_box=(x1, y1, x2, y2),
                padding_ratio=self._padding_ratio,
                min_crop_size=self._min_crop_size,
            )
            crop_left, crop_top, crop_right, crop_bottom = crop_box
            crop_rgb = rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
            mask_area_pixels = int(np.count_nonzero(raw_mask))
            bbox_area = max(1, (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]))

            raw_candidates.append(
                DefectCandidate(
                    label=self.DEFAULT_SEG_LABEL,
                    score=seg_score,
                    summary="stage-1 corrosion segmentation",
                    box={
                        "x1": float(bbox_xyxy[0]),
                        "y1": float(bbox_xyxy[1]),
                        "x2": float(bbox_xyxy[2]),
                        "y2": float(bbox_xyxy[3]),
                    },
                    evidence={
                        "mask_area_pixels": mask_area_pixels,
                        "bbox_area_pixels": bbox_area,
                    },
                )
            )
            detections.append(
                {
                    "bbox_xyxy": bbox_xyxy,
                    "crop_box_xyxy": crop_box,
                    "crop_rgb": crop_rgb,
                    "seg_score": seg_score,
                    "mask_area_pixels": mask_area_pixels,
                    "priority": float(seg_score) * float(max(mask_area_pixels, bbox_area)),
                    "timestamp": now,
                }
            )

        detections = self._attach_tracks(detections, now=now)
        detections.sort(key=lambda item: float(item["seg_score"]), reverse=True)
        detections = detections[: self._max_detections]
        return detections, raw_candidates

    def _attach_tracks(self, detections: list[dict[str, object]], *, now: float) -> list[dict[str, object]]:
        with self._track_lock:
            stale_ids = [
                track_id
                for track_id, track in self._tracks.items()
                if now - float(track.get("last_seen", 0.0)) > self._track_ttl_seconds
            ]
            for track_id in stale_ids:
                self._tracks.pop(track_id, None)

            available_tracks = [
                (track_id, track)
                for track_id, track in self._tracks.items()
                if now - float(track.get("last_seen", 0.0)) <= self._track_ttl_seconds
            ]
            unmatched_track_ids = {track_id for track_id, _track in available_tracks}

            for detection in detections:
                bbox = detection["bbox_xyxy"]
                best_track_id = None
                best_iou = -1.0
                for track_id, track in available_tracks:
                    if track_id not in unmatched_track_ids:
                        continue
                    iou = self._bbox_iou(bbox, track["bbox_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
                if best_track_id is None or best_iou < self._track_match_iou:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    track = {
                        "track_id": track_id,
                        "bbox_xyxy": bbox,
                        "crop_box_xyxy": detection["crop_box_xyxy"],
                        "seg_score": detection["seg_score"],
                        "mask_area_pixels": detection["mask_area_pixels"],
                        "last_seen": now,
                        "cls_label": "",
                        "cls_score": 0.0,
                        "cls_probabilities": {},
                        "cls_updated_at": 0.0,
                        "last_classify_request_at": 0.0,
                        "last_classified_bbox": None,
                        "request_seq": 0,
                        "completed_seq": 0,
                    }
                    self._tracks[track_id] = track
                    detection["track_id"] = track_id
                    detection["track_snapshot"] = dict(track)
                    continue

                unmatched_track_ids.discard(best_track_id)
                track = self._tracks[best_track_id]
                track["bbox_xyxy"] = bbox
                track["crop_box_xyxy"] = detection["crop_box_xyxy"]
                track["seg_score"] = detection["seg_score"]
                track["mask_area_pixels"] = detection["mask_area_pixels"]
                track["last_seen"] = now
                detection["track_id"] = best_track_id
                detection["track_snapshot"] = dict(track)

            for detection in detections:
                track = self._tracks[int(detection["track_id"])]
                detection["track_snapshot"] = dict(track)

        return detections

    def _schedule_classification_jobs(
        self,
        detections: list[dict[str, object]],
    ) -> threading.Event | None:
        if not detections:
            return None

        ranked = sorted(detections, key=lambda item: float(item["priority"]), reverse=True)
        bootstrap_event: threading.Event | None = None
        has_ready_cls = any(
            bool((detection.get("track_snapshot") or {}).get("cls_label"))
            for detection in ranked[: self._classify_top_k]
        )
        now = time.monotonic()

        for detection in ranked[: self._classify_top_k]:
            track_snapshot = dict(detection["track_snapshot"])
            if not self._needs_classification(track_snapshot, detection, now):
                continue

            event = threading.Event()
            with self._track_lock:
                track = self._tracks.get(int(detection["track_id"]))
                if track is None:
                    continue
                if not self._needs_classification(track, detection, now):
                    continue
                track["last_classify_request_at"] = now
                track["request_seq"] = int(track.get("request_seq", 0)) + 1
                request_seq = int(track["request_seq"])

            job = {
                "track_id": int(detection["track_id"]),
                "request_seq": request_seq,
                "crop_rgb": np.ascontiguousarray(detection["crop_rgb"]),
                "bbox_xyxy": tuple(detection["bbox_xyxy"]),
                "event": event,
            }
            try:
                self._job_queue.put_nowait(job)
            except queue.Full:
                break
            if bootstrap_event is None and not has_ready_cls:
                bootstrap_event = event
                has_ready_cls = True
        return bootstrap_event

    def _needs_classification(
        self,
        track: dict[str, object],
        detection: dict[str, object],
        now: float,
    ) -> bool:
        cls_label = str(track.get("cls_label", ""))
        cls_updated_at = float(track.get("cls_updated_at", 0.0))
        last_request_at = float(track.get("last_classify_request_at", 0.0))
        if not cls_label:
            return now - last_request_at >= 0.15
        if now - cls_updated_at >= self._classification_cooldown_seconds:
            return now - last_request_at >= 0.15

        last_bbox = track.get("last_classified_bbox")
        if last_bbox is None:
            return False
        return self._bbox_iou(tuple(detection["bbox_xyxy"]), tuple(last_bbox)) < self._classification_refresh_iou

    def _classification_worker(self) -> None:
        bind_current_thread(
            self._vision_cpuset,
            logger=self.logger,
            label="vision classification worker",
        )
        while True:
            job = self._job_queue.get()
            event = job.get("event")
            try:
                label, score, probabilities = self._classify_crop(job["crop_rgb"])
                with self._track_lock:
                    track = self._tracks.get(int(job["track_id"]))
                    if track is not None and int(job["request_seq"]) >= int(track.get("completed_seq", 0)):
                        track["cls_label"] = label
                        track["cls_score"] = float(score)
                        track["cls_probabilities"] = probabilities
                        track["cls_updated_at"] = time.monotonic()
                        track["completed_seq"] = int(job["request_seq"])
                        track["last_classified_bbox"] = tuple(job["bbox_xyxy"])
            except Exception as exc:
                self.logger.warning("corrosion classifier worker failed: %s", exc)
            finally:
                if isinstance(event, threading.Event):
                    event.set()
                self._job_queue.task_done()

    def _classify_crop(
        self,
        rgb_crop: np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        session = self._ensure_cls_session()
        tensor = self._preprocess_cls_image(rgb_crop)
        raw = session.run([self._cls_output_name], {self._cls_input_name: tensor})[0][0]
        probs = self._ensure_probabilities(raw)
        top1 = int(np.argmax(probs))
        probabilities = {
            class_name: round(float(prob), 6)
            for class_name, prob in zip(self._class_names, probs.tolist())
        }
        return self._class_names[top1], float(probs[top1]), probabilities

    def _build_project_candidates(
        self,
        detections: list[dict[str, object]],
    ) -> list[DefectCandidate]:
        candidates: list[DefectCandidate] = []
        with self._track_lock:
            for detection in detections[: self._max_display_tracks]:
                track = self._tracks.get(int(detection["track_id"]))
                if track is None:
                    continue
                seg_score = float(detection["seg_score"])
                cls_label = str(track.get("cls_label", ""))
                cls_score = float(track.get("cls_score", 0.0))
                if cls_label:
                    label = cls_label
                    score = cls_score
                    summary = (
                        f"Corrosion detected; subtype={cls_label}, "
                        f"seg={seg_score:.2f}, cls={cls_score:.2f}."
                    )
                else:
                    label = self.DEFAULT_SEG_LABEL
                    score = seg_score
                    summary = (
                        f"Corrosion detected; subtype is being refreshed, "
                        f"seg={seg_score:.2f}."
                    )
                bbox = tuple(detection["bbox_xyxy"])
                box = {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                }
                candidates.append(
                    DefectCandidate(
                        label=label,
                        score=score,
                        summary=summary,
                        box=box,
                        evidence={
                            "track_id": int(track["track_id"]),
                            "segmentation_score": round(seg_score, 6),
                            "classification_score": round(cls_score, 6),
                            "mask_area_pixels": int(detection["mask_area_pixels"]),
                            "cls_provider_actual": self._cls_provider_actual,
                            "classification_probabilities": dict(track.get("cls_probabilities", {})),
                        },
                    )
                )
        candidates.sort(
            key=lambda item: (
                0 if item.label == self.DEFAULT_SEG_LABEL else 1,
                float(item.evidence.get("segmentation_score", item.score)),
                item.score,
            ),
            reverse=True,
        )
        return candidates[: self._max_display_tracks]

    def _render_boxes(self, rgb, candidates, output_path):
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for candidate in candidates:
            box = candidate.box
            if not box:
                continue
            x1, y1 = int(box["x1"]), int(box["y1"])
            x2, y2 = int(box["x2"]), int(box["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
            label_text = f"{candidate.label} {candidate.score:.2f}"
            text_width = draw.textlength(label_text, font=font)
            draw.rectangle((x1, max(0, y1 - 22), x1 + text_width + 10, y1), fill=(0, 0, 0))
            draw.text((x1 + 4, max(0, y1 - 20)), label_text, fill=(255, 255, 255), font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb_image(np.asarray(image), output_path)

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not labels:
            raise RuntimeError(f"empty label file: {path}")
        return labels

    @staticmethod
    def _preprocess_cls_image(rgb_crop: np.ndarray, size: int = 224) -> np.ndarray:
        image = Image.fromarray(np.asarray(rgb_crop, dtype=np.uint8), mode="RGB")
        width, height = image.size
        if width < height:
            resized = image.resize((size, int(round(height * size / width))), Image.Resampling.BILINEAR)
        else:
            resized = image.resize((int(round(width * size / height)), size), Image.Resampling.BILINEAR)
        left = max(0, int(round((resized.size[0] - size) / 2.0)))
        top = max(0, int(round((resized.size[1] - size) / 2.0)))
        cropped = resized.crop((left, top, left + size, top + size))
        if cropped.size != (size, size):
            canvas = Image.new("RGB", (size, size))
            canvas.paste(cropped, ((size - cropped.size[0]) // 2, (size - cropped.size[1]) // 2))
            cropped = canvas
        array = np.asarray(cropped, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _ensure_probabilities(output: np.ndarray) -> np.ndarray:
        output = np.asarray(output, dtype=np.float32).reshape(-1)
        looks_like_probs = (
            np.all(output >= -1e-6)
            and np.all(output <= 1.0 + 1e-6)
            and abs(float(output.sum()) - 1.0) <= 1e-3
        )
        if looks_like_probs:
            return output
        shifted = output - np.max(output)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 1e-6:
            return 0.0
        return inter / union

    @staticmethod
    def _build_crop_box_from_mask(
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


class OrtYOLOv8Recognizer(DefectRecognizer):
    """Direct onnxruntime + SpaceMITExecutionProvider YOLOv8n inference.
    No spacemit_vision dependency. Input size is read from the loaded model so
    the same backend can run both 192x192 and 320x320 defect checkpoints.
    Output: [1,7,756] where 7=[cx,cy,w,h,cls0,cls1,cls2]
    Pre/post-processing: pure numpy (no cv2 required).
    """

    MODEL_PATH_DEFAULT = Path("/mnt/ssd/models/vision/defect/yolov8n_corrosion_warmstart_v1.q.onnx")
    INPUT_SIZE = (192, 192)
    CLASSES = ["rust_like_corrosion", "coating_flaking_or_delamination", "chalking_or_powdering"]

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        self._session = None
        self._input_name = None
        self._output_name = None
        self._model_path = None
        self._input_size = self.INPUT_SIZE
        self._score_threshold = 0.25
        self._iou_threshold = 0.45

    @property
    def backend_name(self) -> str:
        return "ort_yolo"

    def analyze_rgb(self, rgb, image_path, annotated_output_path=None):
        session = self._ensure_session()
        src_h, src_w = rgb.shape[:2]
        model_w, model_h = self._input_size

        input_tensor, sx, sy, px, py = self._preprocess_letterbox(rgb, model_w, model_h)

        t0 = time.monotonic()
        outputs = session.run([self._output_name], {self._input_name: input_tensor})
        infer_ms = (time.monotonic() - t0) * 1000

        raw_boxes = self._decode_boxes(outputs[0][0], model_w, model_h, sx, sy, px, py, src_w, src_h)
        candidates = self._build_candidates(raw_boxes)

        annotated_path = None
        if self.config.recognizer.save_annotated_image:
            annotated_path = annotated_output_path or self._default_annotated_path(image_path)
            self._render_boxes(rgb, candidates, annotated_path)

        top = sorted(candidates, key=lambda c: c.score, reverse=True)[:self.config.recognizer.max_candidates]

        return VisionAnalysisResult(
            recognizer_backend=self.backend_name,
            image_path=image_path,
            annotated_image_path=annotated_path,
            candidates=top,
            raw_candidates=candidates[:8],
            metrics={
                "infer_ms": round(infer_ms, 1),
                "detections": len(candidates),
                "model": str(self._model_path or self.MODEL_PATH_DEFAULT),
                "input_size": f"{model_w}x{model_h}",
                "providers": str(session.get_providers()),
            },
            notes=[
                "ort_yolo: onnxruntime + SpaceMIT EP direct YOLOv8n inference",
                f"src {src_w}x{src_h} -> letterbox -> {model_w}x{model_h}, {infer_ms:.0f}ms",
                f"found {len(candidates)} boxes, top {len(top)}",
            ],
        )

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        mp = Path(str(self.config.recognizer.spacemit_model_path or self.MODEL_PATH_DEFAULT))
        if not mp.exists():
            mp = self.MODEL_PATH_DEFAULT
        self.logger.info("loading model: %s", mp)
        import onnxruntime as ort
        self._session = ort.InferenceSession(str(mp), providers=["SpaceMITExecutionProvider"])
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        self._output_name = self._session.get_outputs()[0].name
        self._model_path = mp
        input_shape = list(input_meta.shape)
        if len(input_shape) >= 4:
            model_h = int(input_shape[2])
            model_w = int(input_shape[3])
            self._input_size = (model_w, model_h)
        else:
            self._input_size = self.INPUT_SIZE
        self.logger.info("model loaded. providers=%s", self._session.get_providers())
        return self._session

    @staticmethod
    def _preprocess_letterbox(rgb, target_w, target_h):
        src_h, src_w = rgb.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        image = Image.fromarray(rgb, mode="RGB")
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(image)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        return tensor, scale, scale, float(pad_x), float(pad_y)

    def _decode_boxes(self, output, model_w, model_h, sx, sy, px, py, src_w, src_h):
        boxes_cc = output[:4, :].T
        cls_scores = output[4:, :].T
        max_scores = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)
        keep = max_scores > self._score_threshold
        boxes_cc = boxes_cc[keep]
        scores = max_scores[keep]
        cls_ids = cls_ids[keep]
        if boxes_cc.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        cx, cy, w, h = boxes_cc[:, 0], boxes_cc[:, 1], boxes_cc[:, 2], boxes_cc[:, 3]
        x1 = np.clip((cx - w / 2.0 - px) / sx, 0, src_w)
        y1 = np.clip((cy - h / 2.0 - py) / sy, 0, src_h)
        x2 = np.clip((cx + w / 2.0 - px) / sx, 0, src_w)
        y2 = np.clip((cy + h / 2.0 - py) / sy, 0, src_h)
        boxes = np.stack([x1, y1, x2, y2, scores, cls_ids.astype(np.float32)], axis=1)
        return self._nms(boxes)

    def _nms(self, boxes):
        if boxes.shape[0] <= 1:
            return boxes
        order = boxes[:, 4].argsort()[::-1]
        boxes = boxes[order]
        keep = []
        while boxes.shape[0] > 0:
            keep.append(boxes[0])
            if boxes.shape[0] == 1:
                break
            b0, rest = boxes[0], boxes[1:]
            ix1 = np.maximum(b0[0], rest[:, 0])
            iy1 = np.maximum(b0[1], rest[:, 1])
            ix2 = np.minimum(b0[2], rest[:, 2])
            iy2 = np.minimum(b0[3], rest[:, 3])
            iw = np.maximum(0, ix2 - ix1)
            ih = np.maximum(0, iy2 - iy1)
            inter = iw * ih
            a0 = (b0[2] - b0[0]) * (b0[3] - b0[1])
            areas = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            union = a0 + areas - inter
            iou = inter / np.maximum(union, 1e-6)
            same_cls = rest[:, 5] == b0[5]
            suppress = (iou > self._iou_threshold) & same_cls
            boxes = rest[~suppress]
        return np.stack(keep, axis=0)

    def _build_candidates(self, boxes):
        result = []
        for box in boxes:
            x1, y1, x2, y2, score, cls_id = box
            cls_id = int(cls_id)
            label = self.CLASSES[cls_id] if 0 <= cls_id < len(self.CLASSES) else f"class_{cls_id}"
            result.append(DefectCandidate(
                label=label, score=float(score),
                summary=f"ORT YOLOv8: {label} (conf={score:.3f})",
                box={"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                evidence={"cls_id": cls_id, "backend": "ort_yolo"},
            ))
        return result

    def _render_boxes(self, rgb, candidates, output_path):
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for c in candidates:
            box = c.box
            if not box:
                continue
            x1, y1 = int(box["x1"]), int(box["y1"])
            x2, y2 = int(box["x2"]), int(box["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
            label_text = f"{c.label} {c.score:.2f}"
            tw = draw.textlength(label_text, font=font)
            draw.rectangle((x1, y1 - 22, x1 + tw + 10, y1), fill=(0, 0, 0))
            draw.text((x1 + 4, y1 - 20), label_text, fill=(255, 255, 255), font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb_image(np.asarray(image), output_path)

    def _default_annotated_path(self, image_path):
        return self.config.output_dir / "annotated" / f"{image_path.stem}_ort_yolo.png"


def render_candidate_overlay(rgb, candidates, output_path, header):
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x, y = 18, 18
    lines = [header]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c.label}: {c.score:.2f}")
    lh = 18
    bw = max(draw.textlength(line, font=font) for line in lines) + 16
    bh = lh * len(lines) + 12
    draw.rounded_rectangle((10, 10, 10 + bw, 10 + bh), radius=10, fill=(0, 0, 0))
    for line in lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += lh
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb_image(np.asarray(image), output_path)


def build_defect_recognizer(config, backend_name=None):
    selected = backend_name or config.recognizer.backend
    if selected == "corrosion_two_stage_rt":
        return CorrosionTwoStageRealtimeRecognizer(config)
    if selected == "ort_yolo":
        return OrtYOLOv8Recognizer(config)
    if selected == "heuristic_defect":
        return HeuristicDefectRecognizer(config)
    if selected == "spacemit_vision":
        return SpacemitVisionRecognizer(config)
    raise ValueError(f"Unsupported recognizer backend: {selected}")
