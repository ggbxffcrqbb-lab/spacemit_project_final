from __future__ import annotations

import importlib.util
import logging
import os
import socket
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import numpy as np
import yaml
from contextlib import nullcontext
from pathlib import Path

from app.core.cpu_affinity import bind_current_thread
from app.core.config import AppConfig
from app.vision.camera_backends import build_camera_backend
from app.vision.competition_display import CompetitionDisplay
from app.vision.image_utils import save_rgb_image
from app.vision.recognizer import build_defect_recognizer
from app.vision.status_page import VisionStatusPageWriter


class VisionPipelineService:
    STATUS_SERVER_HOST = "127.0.0.1"
    STATUS_SERVER_PORT = 18765

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger("app.vision.service")
        self.config.vision.output_dir.mkdir(parents=True, exist_ok=True)

    def doctor(self, backend_name: str | None = None, probe: bool = False) -> dict:
        backend = build_camera_backend(self.config.vision, backend_name)
        probe_report = backend.probe(deep=probe)
        recognizer_backend = self.config.vision.recognizer.backend
        runtime_profile = self._build_runtime_profile()
        warnings, errors = self._build_runtime_advice(backend.supports_streaming())
        return {
            "vision_enabled": self.config.vision.enabled,
            "camera_backend_selected": backend.backend_name,
            "recognizer_backend_selected": recognizer_backend,
            "output_dir": str(self.config.vision.output_dir),
            "dependencies": {
                "cam-test": bool(shutil.which("cam-test")),
                "v4l2-ctl": bool(shutil.which("v4l2-ctl")),
                "spacemit_vision": self._module_exists("spacemit_vision"),
                "spacemit_ort": self._module_exists("spacemit_ort"),
                "onnxruntime": self._module_exists("onnxruntime"),
                "numpy": self._module_exists("numpy"),
                "PIL": self._module_exists("PIL"),
            },
            "recognizer_runtime_profile": runtime_profile,
            "warnings": warnings,
            "preflight_errors": errors,
            "probe": probe_report.to_dict(),
        }

    def capture_once(self, backend_name: str | None = None, output_path: str | None = None) -> dict:
        backend = build_camera_backend(self.config.vision, backend_name)
        frame = backend.capture_once(Path(output_path) if output_path else None)
        return {"capture": frame.to_dict()}

    def preview_camera(self, backend_name: str | None = None, dry_run: bool = False) -> dict:
        backend = build_camera_backend(self.config.vision, backend_name)
        command = backend.build_preview_command()
        payload = {
            "backend": backend.backend_name,
            "preview_command": " ".join(command),
            "note": (
                "预览窗口建议在 Muse Pi Pro 本地图形桌面观察。"
                " 仅通过 SSH 触发时，命令可能启动成功，但不一定能在当前终端看到图形窗口。"
            ),
        }
        if dry_run:
            return payload
        backend.launch_preview()
        payload["launched"] = True
        return payload

    def analyze_image(
        self,
        image_path: str,
        recognizer_backend: str | None = None,
        annotated_output_path: str | None = None,
    ) -> dict:
        image = Path(image_path).expanduser()
        if not image.exists():
            raise FileNotFoundError(f"图像文件不存在: {image}")
        recognizer = build_defect_recognizer(self.config.vision, recognizer_backend)
        result = recognizer.analyze(image, Path(annotated_output_path) if annotated_output_path else None)
        return result.to_dict()

    def analyze_camera(
        self,
        camera_backend: str | None = None,
        recognizer_backend: str | None = None,
        capture_output_path: str | None = None,
        annotated_output_path: str | None = None,
    ) -> dict:
        backend = build_camera_backend(self.config.vision, camera_backend)
        recognizer = build_defect_recognizer(self.config.vision, recognizer_backend)
        frame = backend.capture_once(Path(capture_output_path) if capture_output_path else None)
        result = recognizer.analyze(
            frame.image_path,
            Path(annotated_output_path) if annotated_output_path else None,
        )
        return {"capture": frame.to_dict(), "analysis": result.to_dict()}

    def stream_camera(
        self,
        camera_backend: str | None = None,
        recognizer_backend: str | None = None,
        interval_seconds: float = 1.5,
        max_frames: int = 0,
        display_status_page: bool = False,
        performance_mode: bool | None = None,
        display_competition: bool = False,
        analysis_callback: Callable[[dict[str, Any]], None] | None = None,
        assistant_status_getter: Callable[[], dict[str, Any]] | None = None,
        external_stop_event: threading.Event | None = None,
    ) -> dict:
        backend = build_camera_backend(self.config.vision, camera_backend)
        recognizer = build_defect_recognizer(self.config.vision, recognizer_backend)
        warnings, errors = self._build_runtime_advice(for_stream=backend.supports_streaming())
        for warning in warnings:
            self.logger.warning("vision runtime warning: %s", warning)
        if errors:
            joined = "\n".join(f"- {item}" for item in errors)
            raise RuntimeError(f"vision stream preflight failed:\n{joined}")
        interval_seconds = max(0.0, float(interval_seconds))
        stream_config = self.config.vision.stream
        performance_mode = (
            stream_config.performance_mode if performance_mode is None else bool(performance_mode)
        )
        write_latest_capture = stream_config.write_latest_capture and not performance_mode
        write_slot_images = stream_config.write_slot_images and not performance_mode
        generate_status_page = stream_config.generate_status_page and not performance_mode
        startup_skip_analysis_frames = max(0, int(stream_config.startup_skip_analysis_frames))
        competition_analysis_interval_seconds = max(
            0.0,
            float(stream_config.competition_analysis_interval_seconds),
        )
        busy_analysis_interval_seconds = max(
            0.0,
            float(stream_config.busy_analysis_interval_seconds),
        )
        slow_analysis_threshold_seconds = max(
            0.0,
            float(stream_config.slow_analysis_threshold_seconds),
        )
        slow_analysis_cooldown_seconds = max(
            0.0,
            float(stream_config.slow_analysis_cooldown_seconds),
        )
        voice_busy_stages = {
            str(item).strip().lower()
            for item in list(stream_config.voice_busy_stages)
            if str(item).strip()
        }
        voice_stage_intervals = {
            str(stage).strip().lower(): max(0.0, float(interval))
            for stage, interval in dict(stream_config.voice_stage_intervals).items()
            if str(stage).strip()
        }
        voice_pause_stages = {
            str(item).strip().lower()
            for item in list(stream_config.voice_pause_stages)
            if str(item).strip()
        }
        native_video_env = os.getenv("SPACEMIT_COMPETITION_NATIVE_VIDEO", "").strip().lower()
        prefer_native_competition_video = native_video_env in {"1", "true", "yes", "on"}
        default_display_fps = 30.0 if prefer_native_competition_video else 15.0
        try:
            competition_display_fps = max(
                5.0,
                float(
                    os.getenv(
                        "SPACEMIT_COMPETITION_DISPLAY_FPS",
                        str(default_display_fps),
                    )
                ),
            )
        except ValueError:
            competition_display_fps = default_display_fps

        capture_output = self.config.vision.output_dir / "stream_current_capture.jpg"
        capture_slots = [
            self.config.vision.output_dir / "stream_slot_a.jpg",
            self.config.vision.output_dir / "stream_slot_b.jpg",
        ]
        annotated_output = self.config.vision.output_dir / "stream_current_annotated.png"
        status = None
        status_url = ""
        enable_native_preview = bool(display_status_page)
        if display_competition and prefer_native_competition_video:
            enable_native_preview = True
        competition_display = (
            CompetitionDisplay(
                f"{self.config.project_name} Phase 6",
                snapshot_path=self.config.vision.output_dir / "competition_display_snapshot.png",
            )
            if display_competition
            else None
        )
        if generate_status_page:
            status = VisionStatusPageWriter(
                output_dir=self.config.vision.output_dir,
                title=f"{self.config.project_name} Phase 6 Vision Status",
                refresh_seconds=max(1, round(interval_seconds) or 1),
            )
            status.update(
                stage="starting",
                headline="Starting vision stream",
                detail="Capture runs continuously. In competition mode, display refresh and analysis refresh are decoupled.",
                camera_backend=backend.backend_name,
                recognizer_backend=recognizer.backend_name,
                frame_interval_seconds=interval_seconds,
            )
            status_url = self._build_status_page_url(status.html_path)
            self.logger.info("vision status page url: %s", status_url)
            if enable_native_preview:
                self.logger.info("native preview requested; using the same display environment as vision-preview")
            else:
                self.logger.info(
                    "status page launch not requested; stream is running headless and writing outputs to %s",
                    self.config.vision.output_dir,
                )
        elif enable_native_preview:
            self.logger.info("native preview requested; status page is disabled for this run")

        capture_lock = threading.Condition()
        stop_event = external_stop_event or threading.Event()
        latest_packet: dict | None = None
        latest_analysis: dict | None = None
        capture_error: Exception | None = None
        analysis_error: Exception | None = None
        capture_seq = 0
        native_video_hold_lock = threading.Lock()
        native_video_hold_active = False
        native_video_hold_capture_seq = 0
        native_video_hold_until = 0.0
        processed_count = 0
        analyzed_count = 0
        display_seq = 0
        latest_result: dict = {}
        analysis_interval_seconds = max(0.0, interval_seconds)
        if (
            display_competition
            and analysis_interval_seconds <= 0.0
            and competition_analysis_interval_seconds > 0.0
        ):
            analysis_interval_seconds = competition_analysis_interval_seconds
        competition_native_video = False
        last_analysis_wall_seconds = 0.0
        last_analysis_policy = "normal"
        last_capture_policy = "normal"
        degraded_capture_error: Exception | None = None
        degraded_analysis_error: Exception | None = None

        if startup_skip_analysis_frames > 0:
            self.logger.info(
                "vision stream startup stabilization enabled: hiding first %s analysis frames",
                startup_skip_analysis_frames,
            )
        if display_competition and analysis_interval_seconds > 0.0:
            self.logger.info(
                "competition display analysis interval set to %.3fs",
                analysis_interval_seconds,
            )
            self.logger.info(
                "competition display video path: %s (fps=%.1f)",
                (
                    "native gtk preview widget"
                    if prefer_native_competition_video
                    else "software-rendered latest frames"
                ),
                competition_display_fps,
            )

        def capture_worker(stream) -> None:
            nonlocal latest_packet, capture_error, capture_seq
            nonlocal last_capture_policy
            nonlocal native_video_hold_active, native_video_hold_capture_seq
            nonlocal native_video_hold_until
            preview_only_resume_pending = False
            try:
                while not stop_event.is_set():
                    capture_policy = "normal"
                    if (
                        stream is not None
                        and prefer_native_competition_video
                        and assistant_status_getter is not None
                    ):
                        voice_stage = ""
                        try:
                            assistant_status = assistant_status_getter() or {}
                            voice_stage = str(assistant_status.get("voice_stage", "")).strip().lower()
                        except Exception:
                            voice_stage = ""
                        if voice_stage and voice_stage in voice_pause_stages:
                            capture_policy = f"preview_only:{voice_stage}"
                            preview_only_resume_pending = True
                            if capture_policy != last_capture_policy:
                                self.logger.info(
                                    "vision capture policy -> %s (native preview stays live, appsink pulling paused)",
                                    capture_policy,
                                )
                                last_capture_policy = capture_policy
                            time.sleep(0.05)
                            continue
                    if preview_only_resume_pending and stream is not None:
                        restart_for_resume = getattr(stream, "restart_for_resume", None)
                        if callable(restart_for_resume):
                            with native_video_hold_lock:
                                native_video_hold_active = True
                                native_video_hold_capture_seq = capture_seq
                                native_video_hold_until = 0.0
                            self.logger.info(
                                "vision capture policy -> resume_native_capture (restarting usb gst pipeline after preview-only pause)"
                            )
                            restart_for_resume()
                        preview_only_resume_pending = False
                    if capture_policy != last_capture_policy:
                        self.logger.info("vision capture policy -> %s", capture_policy)
                        last_capture_policy = capture_policy

                    started_at = time.perf_counter()
                    next_seq = capture_seq + 1
                    frame = stream.capture_frame(None) if stream is not None else backend.capture_once(None)
                    capture_seconds_value = time.perf_counter() - started_at
                    packet = {
                        "source_frame_index": next_seq,
                        "frame": frame,
                        "capture_seconds": capture_seconds_value,
                    }
                    resume_completed = False
                    with capture_lock:
                        capture_seq = next_seq
                        latest_packet = packet
                        capture_lock.notify_all()
                    with native_video_hold_lock:
                        if native_video_hold_active and next_seq > native_video_hold_capture_seq:
                            native_video_hold_active = False
                            native_video_hold_capture_seq = 0
                            native_video_hold_until = time.perf_counter() + 0.2
                            resume_completed = True
                    if resume_completed:
                        self.logger.info(
                            "vision capture policy -> resume_native_capture_complete (first post-restart frame received)"
                        )
            except Exception as exc:
                with capture_lock:
                    capture_error = exc
                    capture_lock.notify_all()

        def analysis_worker() -> None:
            nonlocal latest_analysis
            nonlocal analysis_error
            nonlocal processed_count
            nonlocal latest_result
            nonlocal capture_seq
            nonlocal analyzed_count
            nonlocal last_analysis_wall_seconds
            nonlocal last_analysis_policy
            bind_current_thread(
                os.getenv("SPACEMIT_VISION_CPUSET", ""),
                logger=self.logger,
                label="vision analysis worker",
            )
            last_analyzed_source_index = 0
            last_analysis_started_at = 0.0
            try:
                while not stop_event.is_set():
                    with capture_lock:
                        while True:
                            has_new_packet = (
                                latest_packet is not None
                                and latest_packet["source_frame_index"] > last_analyzed_source_index
                            )
                            if has_new_packet or capture_error is not None or stop_event.is_set():
                                break
                            capture_lock.wait(timeout=0.2)
                        if stop_event.is_set():
                            return
                        if capture_error is not None:
                            raise capture_error
                        assert latest_packet is not None
                        packet = latest_packet
                    effective_interval_seconds = analysis_interval_seconds
                    current_policy = "normal"
                    voice_stage = ""
                    if assistant_status_getter is not None:
                        try:
                            assistant_status = assistant_status_getter() or {}
                            voice_stage = str(assistant_status.get("voice_stage", "")).strip().lower()
                        except Exception:
                            voice_stage = ""
                        if voice_stage and voice_stage in voice_pause_stages:
                            current_policy = f"voice_pause:{voice_stage}"
                            if current_policy != last_analysis_policy:
                                self.logger.info(
                                    "vision analysis policy -> %s (analysis paused)",
                                    current_policy,
                                )
                                last_analysis_policy = current_policy
                            last_analyzed_source_index = int(packet["source_frame_index"])
                            time.sleep(0.05)
                            continue
                        stage_interval_seconds = voice_stage_intervals.get(voice_stage)
                        if voice_stage and stage_interval_seconds is not None:
                            effective_interval_seconds = max(
                                effective_interval_seconds,
                                stage_interval_seconds,
                            )
                            current_policy = f"voice_busy:{voice_stage}"
                        elif voice_stage and voice_stage in voice_busy_stages:
                            effective_interval_seconds = max(
                                effective_interval_seconds,
                                busy_analysis_interval_seconds,
                            )
                            current_policy = f"voice_busy:{voice_stage}"
                    if (
                        slow_analysis_threshold_seconds > 0.0
                        and slow_analysis_cooldown_seconds > 0.0
                        and last_analysis_wall_seconds >= slow_analysis_threshold_seconds
                    ):
                        effective_interval_seconds = max(
                            effective_interval_seconds,
                            slow_analysis_cooldown_seconds,
                        )
                        if current_policy == "normal":
                            current_policy = (
                                f"slow_recovery:{round(last_analysis_wall_seconds, 3)}s"
                            )
                    if current_policy != last_analysis_policy:
                        self.logger.info(
                            "vision analysis policy -> %s (interval=%.3fs)",
                            current_policy,
                            effective_interval_seconds,
                        )
                        last_analysis_policy = current_policy

                    since_last = time.perf_counter() - last_analysis_started_at
                    if since_last < effective_interval_seconds:
                        time.sleep(min(0.05, effective_interval_seconds - since_last))
                        continue
                    frame = packet["frame"]
                    source_frame_index = int(packet["source_frame_index"])
                    capture_seconds_value = float(packet["capture_seconds"])
                    slot_path = capture_slots[source_frame_index % len(capture_slots)]
                    logical_capture_path = capture_output if write_latest_capture else slot_path
                    if frame.rgb is None:
                        raise RuntimeError("captured frame did not contain in-memory RGB data")
                    if write_slot_images:
                        save_rgb_image(frame.rgb, slot_path)
                    if write_latest_capture:
                        save_rgb_image(frame.rgb, capture_output)
                    latest_display_path = (
                        str(logical_capture_path) if (write_latest_capture or write_slot_images) else ""
                    )

                    started_at = time.perf_counter()
                    last_analysis_started_at = started_at
                    result = recognizer.analyze_rgb(
                        frame.rgb,
                        logical_capture_path,
                        annotated_output if not performance_mode else None,
                    )
                    analysis_seconds_value = time.perf_counter() - started_at
                    last_analysis_wall_seconds = analysis_seconds_value
                    analyzed_count += 1
                    last_analyzed_source_index = source_frame_index

                    if analyzed_count <= startup_skip_analysis_frames:
                        if analyzed_count == startup_skip_analysis_frames:
                            self.logger.info(
                                "vision stream startup stabilization complete after %s hidden analysis frames",
                                startup_skip_analysis_frames,
                            )
                        continue

                    processed_count += 1

                    latest_result = {
                        "capture": {
                            **frame.to_dict(),
                            "image_path": latest_display_path,
                        },
                        "analysis": result.to_dict(),
                        "stream": {
                            "frame_index": processed_count,
                            "source_frame_index": source_frame_index,
                            "captured_frames": capture_seq,
                            "interval_seconds": interval_seconds,
                            "analysis_interval_seconds": analysis_interval_seconds,
                            "capture_seconds": round(capture_seconds_value, 4),
                            "analysis_seconds": round(analysis_seconds_value, 4),
                            "loop_seconds": round(analysis_seconds_value, 4),
                            "analysis_policy": current_policy,
                            "capture_mode": "continuous_gst" if backend.supports_streaming() else "threaded_poll_once",
                            "threading_mode": "capture_thread_plus_inference_thread",
                            "performance_mode": performance_mode,
                            "display_status_page": display_status_page,
                            "display_competition": display_competition,
                            "status_html": str(status.html_path) if status is not None else "",
                            "status_json": str(status.json_path) if status is not None else "",
                            "status_text": str(status.text_path) if status is not None else "",
                            "status_url": status_url,
                        },
                    }
                    latest_analysis = {
                        "result": result,
                        "frame_rgb": frame.rgb,
                        "frame_index": processed_count,
                        "source_frame_index": source_frame_index,
                        "captured_frames": capture_seq,
                        "capture_seconds": capture_seconds_value,
                        "analysis_seconds": analysis_seconds_value,
                        "loop_seconds": analysis_seconds_value,
                        "latest_capture_path": latest_display_path,
                    }
                    if status is not None:
                        status.update(
                            stage="running",
                            headline=f"Vision stream running: analyzed {processed_count} frames",
                            detail=f"Latest analyzed source frame: {source_frame_index}",
                            camera_backend=backend.backend_name,
                            recognizer_backend=recognizer.backend_name,
                            frame_index=processed_count,
                            frame_interval_seconds=analysis_interval_seconds,
                            capture_seconds=capture_seconds_value,
                            analysis_seconds=analysis_seconds_value,
                            loop_seconds=analysis_seconds_value,
                            latest_capture_path=latest_display_path,
                            latest_annotated_path=str(result.annotated_image_path or ""),
                            latest_candidates=[candidate.to_dict() for candidate in result.candidates],
                            latest_metrics=dict(result.metrics),
                            latest_notes=list(result.notes),
                        )
                    if analysis_callback is not None:
                        try:
                            analysis_callback(
                                {
                                    "camera_backend": backend.backend_name,
                                    "recognizer_backend": recognizer.backend_name,
                                    "frame_index": processed_count,
                                    "source_frame_index": source_frame_index,
                                    "captured_frames": capture_seq,
                                    "capture_seconds": capture_seconds_value,
                                    "analysis_seconds": analysis_seconds_value,
                                    "loop_seconds": analysis_seconds_value,
                                    "analysis_policy": current_policy,
                                    "latest_capture_path": latest_display_path,
                                    "result_dict": latest_result["analysis"],
                                }
                            )
                        except Exception as callback_exc:
                            self.logger.warning("vision analysis callback failed: %s", callback_exc)
                    if max_frames > 0 and (display_seq if display_competition else processed_count) >= max_frames:
                        stop_event.set()
                        return
            except Exception as exc:
                with capture_lock:
                    analysis_error = exc
                    capture_lock.notify_all()

        stream_context = (
            backend.open_stream(
                show_preview=enable_native_preview,
                embed_preview=display_competition and prefer_native_competition_video,
            )
            if backend.supports_streaming()
            else nullcontext(None)
        )
        capture_thread = None
        analysis_thread = None
        try:
            with stream_context as stream:
                if (
                    competition_display is not None
                    and prefer_native_competition_video
                    and stream is not None
                    and hasattr(stream, "get_preview_widget")
                ):
                    preview_widget = stream.get_preview_widget()
                    if preview_widget is not None:
                        competition_display.attach_video_widget(preview_widget)
                        competition_native_video = True
                        competition_display.open()
                if stream is not None and hasattr(stream, "start"):
                    stream.start()
                capture_thread = threading.Thread(
                    target=capture_worker,
                    args=(stream,),
                    name="vision_capture_worker",
                    daemon=True,
                )
                capture_thread.start()
                analysis_thread = threading.Thread(
                    target=analysis_worker,
                    name="vision_analysis_worker",
                    daemon=True,
                )
                analysis_thread.start()

                while not stop_event.is_set():
                    with capture_lock:
                        packet = latest_packet
                        cap_err = capture_error
                        infer_err = analysis_error
                    if cap_err is not None:
                        if degraded_capture_error is None:
                            degraded_capture_error = cap_err
                            self.logger.warning(
                                "vision capture degraded, keeping phase6 alive with latest frame: %s",
                                cap_err,
                            )
                            if status is not None:
                                status.update(
                                    stage="degraded",
                                    headline="Vision stream degraded",
                                    detail=(
                                        "Camera stream failed and could not be reopened. "
                                        "Keeping the latest frame and voice interaction alive. "
                                        f"Cause: {type(cap_err).__name__}: {cap_err}"
                                    ),
                                )
                        cap_err = None
                    if infer_err is not None:
                        if degraded_capture_error is not None:
                            if degraded_analysis_error is None:
                                degraded_analysis_error = infer_err
                                self.logger.warning(
                                    "vision analysis worker stopped after capture failure: %s",
                                    infer_err,
                                )
                        else:
                            raise infer_err

                    analysis_snapshot = latest_analysis
                    if packet is None:
                        time.sleep(0.01)
                        continue
                    frame = packet["frame"]
                    capture_seconds_value = float(packet["capture_seconds"])
                    result_obj = analysis_snapshot["result"] if analysis_snapshot is not None else None
                    display_frame_index = int(analysis_snapshot["frame_index"]) if analysis_snapshot else 0
                    display_source_index = (
                        int(analysis_snapshot["source_frame_index"])
                        if analysis_snapshot is not None
                        else int(packet["source_frame_index"])
                    )
                    analysis_seconds_value = float(analysis_snapshot["analysis_seconds"]) if analysis_snapshot else 0.0
                    loop_seconds_value = float(analysis_snapshot["loop_seconds"]) if analysis_snapshot else 0.0

                    if competition_display is not None:
                        display_seq += 1
                        loop_start = time.perf_counter()
                        display_rgb = (
                            frame.rgb if frame is not None
                            else np.zeros((720, 1280, 3), dtype=np.uint8)
                        )
                        if competition_native_video and stream is not None and hasattr(stream, "set_overlay_boxes"):
                            boxes: list[dict] = []
                            if result_obj:
                                for c in (result_obj.candidates or []):
                                    if c.box:
                                        box = dict(c.box)
                                        box["label"] = f"{c.label} {c.score:.2f}"
                                        boxes.append(box)
                            stream.set_overlay_boxes(boxes)
                        assistant_status = (
                            assistant_status_getter() if assistant_status_getter is not None else None
                        )
                        native_video_hold = False
                        if competition_native_video:
                            with native_video_hold_lock:
                                native_video_hold = native_video_hold_active or (
                                    native_video_hold_until > 0.0
                                    and time.perf_counter() < native_video_hold_until
                                )
                        key = competition_display.show(
                            display_rgb,
                            result=result_obj,
                            capture_seconds=capture_seconds_value,
                            analysis_seconds=analysis_seconds_value,
                            loop_seconds=loop_seconds_value,
                            frame_index=display_seq,
                            source_frame_index=display_source_index,
                            captured_frames=capture_seq,
                            camera_backend=backend.backend_name,
                            assistant_status=assistant_status,
                            native_video_hold=native_video_hold,
                        )
                        if key in (27, ord("q"), ord("Q")):
                            if status is not None:
                                status.update(
                                    stage="stopped",
                                    headline="Competition display closed by user",
                                    detail="User pressed ESC/Q on the native competition display window.",
                                )
                            stop_event.set()
                            break
                    elif analysis_snapshot is None:
                        time.sleep(0.01)
                        continue

                    if max_frames > 0 and (display_seq if display_competition else processed_count) >= max_frames:
                        stop_event.set()
                        break
                    if display_competition:
                        elapsed = time.perf_counter() - loop_start
                        target_interval = 1.0 / competition_display_fps
                        sleep_time = max(0.001, target_interval - elapsed)
                        time.sleep(sleep_time)
                    else:
                        time.sleep(max(0.0, interval_seconds))

                stop_event.set()
                if capture_thread is not None:
                    capture_thread.join(timeout=2.0)
                if analysis_thread is not None:
                    analysis_thread.join(timeout=2.0)
        except KeyboardInterrupt:
            if status is not None:
                status.update(
                    stage="stopped",
                    headline="Vision stream stopped",
                    detail=f"Stopped safely after analyzing {processed_count} frames.",
                )
            raise
        except Exception as exc:
            if status is not None:
                status.update(
                    stage="error",
                    headline="Vision stream failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            raise
        finally:
            stop_event.set()
            if capture_thread is not None and capture_thread.is_alive():
                capture_thread.join(timeout=2.0)
            if analysis_thread is not None and analysis_thread.is_alive():
                analysis_thread.join(timeout=2.0)
            if competition_display is not None:
                competition_display.close()

        if degraded_capture_error is not None:
            latest_result.setdefault("stream", {})
            latest_result["stream"]["degraded"] = True
            latest_result["stream"]["degraded_reason"] = (
                f"{type(degraded_capture_error).__name__}: {degraded_capture_error}"
            )

        return latest_result

    def _build_status_page_url(self, status_html_path: Path) -> str:
        serve_root = status_html_path.parent.parent
        try:
            relative_path = status_html_path.relative_to(serve_root).as_posix()
        except ValueError:
            serve_root = status_html_path.parent
            relative_path = status_html_path.name
        base_url = self._ensure_status_http_server(serve_root)
        return f"{base_url}/{relative_path}"

    def _ensure_status_http_server(self, serve_root: Path) -> str:
        host = self.STATUS_SERVER_HOST
        port = self.STATUS_SERVER_PORT
        if not self._port_open(host, port):
            cmd = [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                host,
                "--directory",
                str(serve_root),
            ]
            self.logger.info("launch status http server: %s", " ".join(cmd))
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            for _ in range(20):
                if self._port_open(host, port):
                    break
                time.sleep(0.1)
        return f"http://{host}:{port}"

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            return False

    @staticmethod
    def _module_exists(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def _build_runtime_profile(self) -> dict:
        profile = {
            "backend": self.config.vision.recognizer.backend,
            "spacemit_vision_config": str(self.config.vision.recognizer.spacemit_vision_config or ""),
            "model_path": "",
            "label_file_path": "",
            "providers": [],
            "uses_spacemit_ep": False,
            "model_is_quantized_q_onnx": False,
        }
        if self.config.vision.recognizer.backend != "spacemit_vision":
            return profile
        config_path = self.config.vision.recognizer.spacemit_vision_config
        if not config_path:
            return profile
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        except Exception as exc:
            profile["read_error"] = f"{type(exc).__name__}: {exc}"
            return profile

        default_params = raw.get("default_params", {}) or {}
        providers = default_params.get("providers", []) or []
        model_path = str(raw.get("model_path", "") or "")
        label_file_path = str(raw.get("label_file_path", "") or "")
        normalized_model_path = model_path.lower()
        uses_spacemit_ep = any(str(item) == "SpaceMITExecutionProvider" for item in providers)

        profile.update(
            {
                "model_path": model_path,
                "label_file_path": label_file_path,
                "providers": [str(item) for item in providers],
                "uses_spacemit_ep": uses_spacemit_ep,
                "model_is_quantized_q_onnx": normalized_model_path.endswith(".q.onnx"),
            }
        )
        return profile

    def _build_runtime_advice(self, for_stream: bool) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        profile = self._build_runtime_profile()

        if self._module_exists("onnxruntime") and self._module_exists("spacemit_ort"):
            try:
                import onnxruntime as _ort
                _ver = getattr(_ort, "__version__", "")
            except Exception:
                _ver = ""
            if "spacemit" not in str(_ver).lower():
                warnings.append(
                    "当前环境同时存在通用版 onnxruntime 与 spacemit-ort。"
                    "官方 FAQ 明确提示两者混装可能导致板端推理冲突，建议尽快清理为单一正式运行时。"
                )

        if profile["backend"] != "spacemit_vision":
            return warnings, errors

        model_path = str(profile.get("model_path", ""))
        label_file_path = str(profile.get("label_file_path", ""))
        uses_spacemit_ep = bool(profile.get("uses_spacemit_ep"))
        is_quantized = bool(profile.get("model_is_quantized_q_onnx"))

        if model_path.endswith(".onnx") and not model_path.endswith(".q.onnx"):
            warnings.append(
                f"当前 spacemit_vision 模型是普通 ONNX：{model_path}。官方 YOLOv8/Model Zoo 默认板端部署形态是 q.onnx 量化模型。"
            )
        if "coco.txt" in label_file_path.replace("\\", "/").lower():
            warnings.append(
                "当前识别标签文件仍是 COCO 通用标签，说明这套配置更适合验证官方视觉链路，不适合作为最终防腐缺陷检测结果口径。"
            )
        if for_stream and uses_spacemit_ep and model_path and not is_quantized:
            errors.append(
                "当前连续流配置使用 SpaceMITExecutionProvider + 非 q.onnx 模型。该组合在板端已实测触发 `tcm buffer alloc failed`，请先在 x86 环境按官方 xquant 流程量化，再把 q.onnx 同步到 Muse Pi Pro。"
            )
        return warnings, errors
