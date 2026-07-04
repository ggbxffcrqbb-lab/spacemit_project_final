from __future__ import annotations

import abc
import json
import logging
import threading
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from app.core.config import VisionConfig
from app.vision.image_utils import load_nv12_as_rgb, nv12_bytes_to_rgb, save_rgb_image
from app.vision.types import CameraProbeReport, CapturedFrame

if TYPE_CHECKING:
    from gi.repository import Gst


AUTO_JSON_PATTERN = re.compile(r"save json to\s+(?P<path>\S+\.json)\s+success", re.IGNORECASE)
DETECTED_SENSOR_PATTERN = re.compile(
    r"detect\s+(?P<sensor>[\w.-]+)\s+sensors\s+in\s+csi\d+:\s+success",
    re.IGNORECASE,
)
CONFIG_SENSOR_PATTERN = re.compile(r"sensor_name:\s*(?P<sensor>[\w.-]+)")
NV12_DUMP_PATTERN = re.compile(r"dump cpp output image to\s+(?P<path>\S+\.nv12)")
RAW_DUMP_PATTERN = re.compile(r"dump raw output image to\s+(?P<path>\S+\.raw)")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class CameraBackend(abc.ABC):
    def __init__(self, config: VisionConfig):
        self.config = config
        self.logger = logging.getLogger(f"app.vision.camera.{self.backend_name}")

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def probe(self, deep: bool = False) -> CameraProbeReport:
        raise NotImplementedError

    @abc.abstractmethod
    def capture_once(self, output_path: Path | None = None) -> CapturedFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def build_preview_command(self) -> list[str]:
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        return False

    def open_stream(self, show_preview: bool = False, embed_preview: bool = False):
        raise NotImplementedError(f"{self.backend_name} does not implement continuous streaming")

    def launch_preview(self) -> dict[str, str]:
        command = self.build_preview_command()
        self.logger.info("launch preview command: %s", " ".join(command))
        subprocess.run(command, check=False, env=self.preview_env())
        return {
            "backend": self.backend_name,
            "preview_command": " ".join(command),
        }

    def preview_env(self) -> dict[str, str]:
        return os.environ.copy()

    def _run(self, args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        self.logger.info("run command: %s", " ".join(args))
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )


class CameraFrameStream(abc.ABC):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @abc.abstractmethod
    def capture_frame(self, output_path: Path | None = None) -> CapturedFrame:
        raise NotImplementedError

    def get_preview_widget(self):
        if getattr(self, "_preview_widget", None) is not None:
            return self._preview_widget
        if getattr(self, "_preview_sink", None) is None:
            return None
        try:
            return self._preview_sink.get_property("widget")
        except Exception:
            return None

    def start(self) -> None:
        return None

    def detach_preview_widget(self) -> None:
        widget = self.get_preview_widget()
        if widget is None:
            return
        try:
            parent = widget.get_parent()
        except Exception:
            parent = None
        if parent is not None and hasattr(parent, "remove"):
            try:
                parent.remove(widget)
            except Exception:
                pass
        if hasattr(widget, "hide"):
            try:
                widget.hide()
            except Exception:
                pass

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError


@dataclass
class UsbDeviceInfo:
    card_label: str
    device_path: Path


class OfficialMipiGstFrameStream(CameraFrameStream):
    def __init__(
        self,
        backend: "OfficialMipiCameraBackend",
        *,
        auto_json: Path,
        sensor: str,
        autostart: bool = True,
    ):
        self.backend = backend
        self.logger = logging.getLogger("app.vision.camera.mipi_official.stream")
        self.auto_json = auto_json
        self.sensor = sensor
        self.frame_index = 0
        self._started = False
        self._gst = self._load_gst()
        self._pipeline = self._build_pipeline()
        self._sink = self._pipeline.get_by_name("appsink0")
        self._preview_sink = self._pipeline.get_by_name("previewsink0")
        self._preview_widget = None
        if self._preview_sink is not None:
            try:
                self._preview_widget = self._preview_sink.get_property("widget")
            except Exception:
                self._preview_widget = None
        if self._sink is None:
            raise RuntimeError("GStreamer appsink 初始化失败")
        self._bus = self._pipeline.get_bus()
        if autostart:
            self.start()

    def start(self) -> None:
        if self._started:
            return
        state_change = self._pipeline.set_state(self._gst.State.PLAYING)
        self.logger.info("mipi gst stream state change: %s", state_change.value_nick)
        self._started = True

    def capture_frame(self, output_path: Path | None = None) -> CapturedFrame:
        sample = self._pull_sample(timeout_seconds=max(2, self.backend.config.mipi.capture_timeout_seconds))
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        pixel_format = str(structure.get_value("format"))

        buffer = sample.get_buffer()
        ok, info = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("GStreamer buffer 映射失败")

        try:
            stride = self._infer_stride(info.size, width, height)
            rgb = nv12_bytes_to_rgb(info.data, width=width, height=height, stride=stride)
        finally:
            buffer.unmap(info)

        output_path = output_path or self.backend._default_output_path(".png")
        save_rgb_image(rgb, output_path)
        self.frame_index += 1
        return CapturedFrame(
            backend=self.backend.backend_name,
            image_path=output_path,
            width=width,
            height=height,
            pixel_format=pixel_format,
            sensor=self.sensor,
            rgb=rgb,
            details={
                "auto_json": str(self.auto_json),
                "capture_mode": "gst_apps_sink",
                "stream_frame_index": self.frame_index,
                "stride": stride,
            },
        )

    def set_overlay_boxes(self, boxes: list[dict]) -> None:
        with self._overlay_lock:
            self._overlay_boxes = list(boxes)

    def _on_cairo_draw(self, overlay, cr, timestamp, duration):
        with self._overlay_lock:
            boxes = list(self._overlay_boxes)
        if not boxes:
            return
        for box in boxes:
            x1 = int(box.get("x1", 0))
            y1 = int(box.get("y1", 0))
            x2 = int(box.get("x2", 0))
            y2 = int(box.get("y2", 0))
            if x2 <= x1 or y2 <= y1:
                continue
            cr.set_source_rgba(0.0, 1.0, 0.0, 0.85)
            cr.set_line_width(3.0)
            cr.rectangle(x1, y1, x2 - x1, y2 - y1)
            cr.stroke()
            label = str(box.get("label", ""))
            if label:
                try:
                    cr.set_font_size(16)
                    extents = cr.text_extents(label)
                    pad = 4
                    cr.set_source_rgba(0.0, 0.0, 0.0, 0.75)
                    cr.rectangle(x1, y1 - extents.height - pad * 2, extents.width + pad * 2, extents.height + pad * 2)
                    cr.fill()
                    cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                    cr.move_to(x1 + pad, y1 - pad)
                    cr.show_text(label)
                except Exception:
                    pass

    def close(self) -> None:
        self.detach_preview_widget()
        self._pipeline.set_state(self._gst.State.NULL)

    def _build_pipeline(self):
        return self._gst.parse_launch(
            " ".join(
                [
                    "spacemitsrc",
                    f"location={self.auto_json}",
                    "close-dmabuf=1",
                    "drop-frames=1",
                    "num-capture-buffers=4",
                    "timeout=500",
                    "!",
                    "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1",
                    "!",
                    "appsink",
                    "name=appsink0",
                    "emit-signals=false",
                    "sync=false",
                    "drop=true",
                    "max-buffers=1",
                    "wait-on-eos=false",
                ]
            )
        )

    def _pull_sample(self, timeout_seconds: int):
        deadline = time.monotonic() + timeout_seconds
        last_warning = ""
        while time.monotonic() < deadline:
            sample = self._sink.emit("try-pull-sample", 200 * self._gst.MSECOND)
            if sample is not None:
                return sample
            last_warning = self._drain_bus_messages()
        detail = f"，最后一条总线消息: {last_warning}" if last_warning else ""
        raise RuntimeError(f"MIPI 连续取流超时，未收到新帧{detail}")

    def _drain_bus_messages(self) -> str:
        last_message = ""
        while True:
            message = self._bus.pop_filtered(
                self._gst.MessageType.ERROR | self._gst.MessageType.WARNING | self._gst.MessageType.EOS
            )
            if message is None:
                return last_message
            source_name = message.src.get_name() if message.src else "unknown"
            if message.type == self._gst.MessageType.ERROR:
                err, debug = message.parse_error()
                raise RuntimeError(f"GStreamer ERROR from {source_name}: {err}; debug={debug}")
            if message.type == self._gst.MessageType.EOS:
                raise RuntimeError(f"GStreamer 提前收到 EOS: {source_name}")
            warn, debug = message.parse_warning()
            last_message = f"{source_name}: {warn}; debug={debug}"
            self.logger.warning("gstreamer warning from %s: %s; debug=%s", source_name, warn, debug)

    @staticmethod
    def _infer_stride(buffer_size: int, width: int, height: int) -> int:
        stride = (buffer_size * 2) // (height * 3)
        return max(width, stride)

    @staticmethod
    def _load_gst():
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return Gst


class OfficialMipiCameraBackend(CameraBackend):
    @property
    def backend_name(self) -> str:
        return "mipi_official"

    def supports_streaming(self) -> bool:
        return True

    def open_stream(self, show_preview: bool = False, embed_preview: bool = False) -> CameraFrameStream:
        probe = self.probe(deep=True)
        if not probe.ok:
            raise RuntimeError(probe.message)
        return OfficialMipiGstFrameStream(
            self,
            auto_json=Path(str(probe.details["auto_json"])),
            sensor=str(probe.details.get("sensor", "")),
            autostart=not embed_preview,
        )

    def probe(self, deep: bool = False) -> CameraProbeReport:
        cam_test = shutil.which("cam-test")
        if not cam_test:
            return CameraProbeReport(
                backend=self.backend_name,
                ok=False,
                message="未找到 cam-test，官方 MIPI 采集链不可用",
            )

        if not deep:
            return CameraProbeReport(
                backend=self.backend_name,
                ok=True,
                message="cam-test 已就绪，未执行深度探测",
                details={
                    "cam_test": cam_test,
                    "detect_json_candidates": [str(path) for path in self.config.mipi.detect_json_candidates],
                },
            )

        cached_report = self._probe_from_cached_auto_json(cam_test)
        if cached_report is not None:
            return cached_report

        candidate_reports: list[dict[str, str | int | bool]] = []
        for detect_json in self.config.mipi.detect_json_candidates:
            if not detect_json.exists():
                candidate_reports.append(
                    {
                        "detect_json": str(detect_json),
                        "ok": 0,
                        "message": "detect json 不存在",
                    }
                )
                continue

            result = self._run([cam_test, str(detect_json)], self.config.mipi.capture_timeout_seconds)
            output = self._sanitize_output(result.stdout or "")
            auto_json = self._extract_match(AUTO_JSON_PATTERN, output)
            auto_json_path = self._resolve_auto_json_path(detect_json, auto_json)
            sensor = self._extract_sensor(output)
            if auto_json_path and auto_json_path.exists():
                sensor = self._load_sensor_from_auto_json(auto_json_path) or sensor

            candidate_reports.append(
                {
                    "detect_json": str(detect_json),
                    "returncode": result.returncode,
                    "sensor": sensor,
                    "auto_json": str(auto_json_path) if auto_json_path else "",
                    "cached": False,
                }
            )

            if auto_json_path and auto_json_path.exists():
                return CameraProbeReport(
                    backend=self.backend_name,
                    ok=True,
                    message="官方 MIPI 探测成功",
                    details={
                        "cam_test": cam_test,
                        "detect_json": str(detect_json),
                        "auto_json": str(auto_json_path),
                        "sensor": sensor,
                        "attempts": candidate_reports,
                    },
                )

        return CameraProbeReport(
            backend=self.backend_name,
            ok=False,
            message="未能通过官方 detect json 探测到可用的 MIPI 传感器",
            details={"cam_test": cam_test, "attempts": candidate_reports},
        )

    def capture_once(self, output_path: Path | None = None) -> CapturedFrame:
        probe = self.probe(deep=True)
        if not probe.ok:
            raise RuntimeError(probe.message)

        cam_test = shutil.which("cam-test")
        auto_json = Path(str(probe.details["auto_json"]))
        capture_started_at = time.time()
        timed_out = False
        try:
            result = self._run([cam_test, str(auto_json)], self.config.mipi.capture_timeout_seconds)
            output = self._sanitize_output(result.stdout or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            output = self._sanitize_output(
                self._coerce_text(exc.stdout) + self._coerce_text(exc.stderr)
            )
            result = None

        nv12_dump = self._extract_match(NV12_DUMP_PATTERN, output)
        raw_dump = self._extract_match(RAW_DUMP_PATTERN, output)
        if not nv12_dump:
            latest_nv12 = self._find_latest_generated_file("cpp0_output_*_s*.nv12", capture_started_at)
            nv12_dump = str(latest_nv12) if latest_nv12 else ""
        if not raw_dump:
            latest_raw = self._find_latest_generated_file("raw_output*.raw", capture_started_at)
            raw_dump = str(latest_raw) if latest_raw else ""
        if not nv12_dump:
            if timed_out:
                raise RuntimeError("官方 MIPI 采集超时，且未找到可用的 NV12 dump 文件")
            if result is not None and result.returncode != 0:
                raise RuntimeError(f"cam-test 运行失败: {output.strip()}")
            raise RuntimeError("官方 MIPI 采集完成，但未找到 NV12 dump 文件")

        nv12_path = Path(nv12_dump)
        if not nv12_path.exists():
            raise RuntimeError(f"NV12 dump 文件不存在: {nv12_path}")

        rgb = load_nv12_as_rgb(nv12_path)
        output_path = output_path or self._default_output_path(".png")
        save_rgb_image(rgb, output_path)

        if not self.config.keep_capture_artifacts:
            self._cleanup(nv12_path)
            if raw_dump:
                self._cleanup(Path(raw_dump))

        height, width = rgb.shape[:2]
        return CapturedFrame(
            backend=self.backend_name,
            image_path=output_path,
            width=width,
            height=height,
            pixel_format="RGB",
            sensor=str(probe.details.get("sensor", "")),
            rgb=rgb,
            details={
                "auto_json": str(auto_json),
                "source_nv12_dump": str(nv12_path),
                "source_raw_dump": raw_dump or "",
                "command_timed_out": timed_out,
                "command_returncode": result.returncode if result is not None else 124,
            },
        )

    def build_preview_command(self) -> list[str]:
        probe = self.probe(deep=True)
        if not probe.ok:
            raise RuntimeError(probe.message)

        gst_launch = shutil.which("gst-launch-1.0")
        auto_json = str(probe.details.get("auto_json", ""))
        if not gst_launch or not auto_json:
            raise RuntimeError("未能生成可用的官方 MIPI 预览命令")

        return [
            gst_launch,
            "-e",
            "spacemitsrc",
            f"location={auto_json}",
            "close-dmabuf=1",
            "!",
            "video/x-raw,format=NV12,width=1920,height=1080",
            "!",
            "videoconvert",
            "!",
            "waylandsink",
            "sync=0",
            "render-rectangle=<0,0,1280,720>",
        ]

    def preview_env(self) -> dict[str, str]:
        env = super().preview_env()
        env.setdefault("DISPLAY", ":0")
        uid = os.getuid()
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        env.setdefault("WAYLAND_DISPLAY", "wayland-0")
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
        env.setdefault("XDG_SESSION_TYPE", "wayland")
        return env

    def _default_output_path(self, suffix: str) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return self.config.output_dir / "captures" / f"mipi_{timestamp}{suffix}"

    @staticmethod
    def _extract_match(pattern: re.Pattern[str], output: str) -> str:
        match = pattern.search(output)
        return match.group("path" if "path" in pattern.groupindex else "sensor") if match else ""

    def _resolve_auto_json_path(self, detect_json: Path, parsed_path: str) -> Path | None:
        if parsed_path:
            return Path(parsed_path)
        candidate_name = detect_json.name.replace("_detect.json", "_auto.json")
        candidate = self.config.mipi.auto_json_dir / candidate_name
        return candidate if candidate.exists() else None

    def _probe_from_cached_auto_json(self, cam_test: str) -> CameraProbeReport | None:
        for detect_json in self.config.mipi.detect_json_candidates:
            auto_json_path = self._resolve_auto_json_path(detect_json, "")
            if auto_json_path is None or not auto_json_path.exists():
                continue
            sensor = self._load_sensor_from_auto_json(auto_json_path)
            if self.config.mipi.prefer_sensor and sensor and sensor != self.config.mipi.prefer_sensor:
                continue
            return CameraProbeReport(
                backend=self.backend_name,
                ok=True,
                message="复用已缓存的官方 auto json",
                details={
                    "cam_test": cam_test,
                    "detect_json": str(detect_json),
                    "auto_json": str(auto_json_path),
                    "sensor": sensor,
                    "attempts": [
                        {
                            "detect_json": str(detect_json),
                            "auto_json": str(auto_json_path),
                            "sensor": sensor,
                            "cached": True,
                        }
                    ],
                },
            )
        return None

    @staticmethod
    def _extract_sensor(output: str) -> str:
        detected = DETECTED_SENSOR_PATTERN.search(output)
        if detected:
            return detected.group("sensor")
        configured = CONFIG_SENSOR_PATTERN.search(output)
        if configured:
            return configured.group("sensor")
        return ""

    @staticmethod
    def _load_sensor_from_auto_json(auto_json_path: Path) -> str:
        try:
            payload = json.loads(auto_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        for value in payload.values():
            if isinstance(value, dict):
                sensor_name = value.get("sensor_name")
                if isinstance(sensor_name, str) and sensor_name:
                    return sensor_name
        return ""

    @staticmethod
    def _cleanup(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _find_latest_generated_file(self, pattern: str, started_at: float) -> Path | None:
        candidates = sorted(
            self.config.mipi.capture_tmp_dir.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                stat = candidate.stat()
            except OSError:
                continue
            if stat.st_mtime >= started_at - 1.0 and stat.st_size > 0:
                return candidate
        return None

    @staticmethod
    def _sanitize_output(output: str) -> str:
        return ANSI_ESCAPE_PATTERN.sub("", output).replace("\r", "")

    @staticmethod
    def _coerce_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value


class UsbV4L2CameraBackend(CameraBackend):
    @property
    def backend_name(self) -> str:
        return "usb_v4l2"

    def supports_streaming(self) -> bool:
        return True

    def open_stream(self, show_preview: bool = False, embed_preview: bool = False) -> CameraFrameStream:
        probe = self.probe(deep=True)
        if not probe.ok:
            raise RuntimeError(probe.message)
        return UsbMjpgGstFrameStream(
            self,
            device=Path(str(probe.details["device"])),
            width=int(probe.details["width"]),
            height=int(probe.details["height"]),
            show_preview=show_preview,
            autostart=not embed_preview,
        )

    def probe(self, deep: bool = False) -> CameraProbeReport:
        v4l2_ctl = shutil.which("v4l2-ctl")
        if not v4l2_ctl:
            return CameraProbeReport(
                backend=self.backend_name,
                ok=False,
                message="未找到 v4l2-ctl，USB 相机后端不可用",
            )
        device_path, device_details = self._resolve_device(v4l2_ctl)
        if not device_path.exists():
            return CameraProbeReport(
                backend=self.backend_name,
                ok=False,
                message=f"USB 相机设备不存在: {device_path}",
            )

        details = {
            "v4l2_ctl": v4l2_ctl,
            "device": str(device_path),
            "auto_detected": device_details.get("auto_detected", False),
            "card_label": device_details.get("card_label", ""),
            "pixel_format": self.config.usb.pixel_format,
            "width": self.config.usb.width,
            "height": self.config.usb.height,
        }
        if deep:
            result = self._run(
                [v4l2_ctl, "--device", str(device_path), "--list-formats-ext"],
                self.config.usb.capture_timeout_seconds,
            )
            details["formats"] = result.stdout
            return CameraProbeReport(
                backend=self.backend_name,
                ok=result.returncode == 0,
                message="USB 相机探测完成" if result.returncode == 0 else "USB 相机探测失败",
                details=details,
            )

        return CameraProbeReport(
            backend=self.backend_name,
            ok=True,
            message="USB 相机设备节点存在，未执行深度探测",
            details=details,
        )

    def capture_once(self, output_path: Path | None = None) -> CapturedFrame:
        probe = self.probe(deep=False)
        if not probe.ok:
            raise RuntimeError(probe.message)
        if self.config.usb.pixel_format.upper() != "MJPG":
            raise RuntimeError("当前 USB 后端仅实现了 MJPG 单帧采集，后续再扩展 YUYV/NV12")

        v4l2_ctl = str(probe.details["v4l2_ctl"])
        device = str(probe.details["device"])
        temp_path = self.config.output_dir / "captures" / "usb_capture_tmp.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        result = self._run(
            [
                v4l2_ctl,
                "--device",
                device,
                f"--set-fmt-video=width={self.config.usb.width},height={self.config.usb.height},pixelformat=MJPG",
                "--stream-mmap=3",
                "--stream-count=1",
                f"--stream-to={temp_path}",
            ],
            self.config.usb.capture_timeout_seconds,
        )
        if result.returncode != 0 or not temp_path.exists():
            raise RuntimeError(f"USB 相机采集失败: {result.stdout.strip()}")

        output_path = output_path or self._default_output_path(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(temp_path) as image:
            rgb = image.convert("RGB")
            rgb.save(output_path)
            width, height = rgb.size

        if not self.config.keep_capture_artifacts:
            temp_path.unlink(missing_ok=True)

        return CapturedFrame(
            backend=self.backend_name,
            image_path=output_path,
            width=width,
            height=height,
            pixel_format="RGB",
            rgb=np.asarray(rgb),
            details={"device": device, "pixel_format": self.config.usb.pixel_format},
        )

    def build_preview_command(self) -> list[str]:
        probe = self.probe(deep=False)
        if not probe.ok:
            raise RuntimeError(probe.message)

        device = str(probe.details["device"])
        width = str(self.config.usb.width)
        height = str(self.config.usb.height)
        gst_launch = shutil.which("gst-launch-1.0")
        if gst_launch:
            return [
                gst_launch,
                "v4l2src",
                f"device={device}",
                "!",
                f"image/jpeg,width={width},height={height},framerate=30/1",
                "!",
                "jpegparse",
                "!",
                "spacemitdec",
                "!",
                "videoconvert",
                "!",
                "autovideosink",
                "sync=0",
            ]

        ffplay = shutil.which("ffplay")
        pixel_format = self.config.usb.pixel_format.lower()
        if ffplay:
            return [
                ffplay,
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-framedrop",
                "-window_title",
                "spacemit_project_usb_preview",
                "-f",
                "video4linux2",
                "-input_format",
                pixel_format,
                "-video_size",
                f"{width}x{height}",
                device,
            ]

        raise RuntimeError("未找到 USB 预览工具，请安装 ffplay 或 gst-launch-1.0")

    def preview_env(self) -> dict[str, str]:
        env = super().preview_env()
        env.setdefault("DISPLAY", ":0")
        uid = os.getuid()
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        env.setdefault("WAYLAND_DISPLAY", "wayland-0")
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
        env.setdefault("XDG_SESSION_TYPE", "wayland")
        return env

    def _default_output_path(self, suffix: str) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return self.config.output_dir / "captures" / f"usb_{timestamp}{suffix}"

    def resolve_stream_device(self, fallback_device: Path | None = None) -> Path:
        configured_path = self._prefer_stable_device_path(Path(self.config.usb.device))
        v4l2_ctl = shutil.which("v4l2-ctl")
        if not v4l2_ctl:
            return fallback_device or configured_path

        if self.config.usb.auto_detect:
            candidate = self._select_candidate_device(v4l2_ctl)
            if candidate is not None:
                return self._prefer_stable_device_path(candidate.device_path)

        if configured_path.exists() and self._device_supports_pixel_format(v4l2_ctl, configured_path):
            return configured_path
        return fallback_device or configured_path

    def _resolve_device(self, v4l2_ctl: str) -> tuple[Path, dict[str, object]]:
        configured_path = Path(self.config.usb.device)
        if not self.config.usb.auto_detect:
            return self._prefer_stable_device_path(configured_path), {
                "auto_detected": False,
                "card_label": "",
            }

        candidate = self._select_candidate_device(v4l2_ctl)
        if candidate is not None:
            stable_path = self._prefer_stable_device_path(candidate.device_path)
            return stable_path, {
                    "auto_detected": True,
                    "card_label": candidate.card_label,
                    "device_node": str(candidate.device_path),
                }

        return self._prefer_stable_device_path(configured_path), {
            "auto_detected": False,
            "card_label": "",
        }

    def _select_candidate_device(self, v4l2_ctl: str) -> UsbDeviceInfo | None:
        candidates = self._list_usb_devices(v4l2_ctl)
        preferred_patterns = [pattern for pattern in self.config.usb.preferred_device_patterns if pattern]
        if preferred_patterns:
            regexes = [re.compile(pattern, re.IGNORECASE) for pattern in preferred_patterns]
            preferred = [
                device
                for device in candidates
                if any(
                    regex.search(device.card_label) or regex.search(str(device.device_path))
                    for regex in regexes
                )
            ]
            if preferred:
                candidates = preferred

        for candidate in candidates:
            if self._device_supports_pixel_format(v4l2_ctl, candidate.device_path):
                return candidate
        return None

    def _list_usb_devices(self, v4l2_ctl: str) -> list[UsbDeviceInfo]:
        result = self._run([v4l2_ctl, "--list-devices"], max(4, self.config.usb.capture_timeout_seconds))
        if result.returncode != 0:
            return []

        devices: list[UsbDeviceInfo] = []
        current_label = ""
        current_is_usb = False
        for raw_line in result.stdout.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                current_label = ""
                current_is_usb = False
                continue
            if not line.startswith("\t"):
                current_label = line.rstrip(":")
                current_is_usb = "usb-" in current_label.lower() or "uvc" in current_label.lower()
                continue
            if not current_is_usb:
                continue
            device = line.strip()
            if not device.startswith("/dev/video"):
                continue
            devices.append(UsbDeviceInfo(card_label=current_label, device_path=Path(device)))
        return devices

    def _device_supports_pixel_format(self, v4l2_ctl: str, device_path: Path) -> bool:
        result = self._run(
            [v4l2_ctl, "--device", str(device_path), "--list-formats-ext"],
            max(4, self.config.usb.capture_timeout_seconds),
        )
        return result.returncode == 0 and self.config.usb.pixel_format.upper() in result.stdout.upper()

    @staticmethod
    def _prefer_stable_device_path(device_path: Path) -> Path:
        aliases: list[Path] = []
        try:
            device_real = device_path.resolve()
        except OSError:
            return device_path

        for root_name in ("by-id", "by-path"):
            root = Path("/dev/v4l") / root_name
            if not root.exists():
                continue
            try:
                entries = sorted(root.iterdir(), key=lambda item: item.name.lower())
            except OSError:
                continue
            for alias in entries:
                if not alias.is_symlink():
                    continue
                try:
                    alias_real = alias.resolve()
                except OSError:
                    continue
                if alias_real != device_real:
                    continue
                aliases.append(alias)

        if not aliases:
            return device_path

        def alias_sort_key(alias: Path) -> tuple[int, int, str]:
            name = alias.name.lower()
            root_rank = 0 if alias.parent.name == "by-id" else 1
            if "video-index0" in name:
                index_rank = 0
            elif "video-index1" in name:
                index_rank = 1
            else:
                index_rank = 2
            return (root_rank, index_rank, name)

        return min(aliases, key=alias_sort_key)


class UsbMjpgGstFrameStream(CameraFrameStream):
    def __init__(
        self,
        backend: "UsbV4L2CameraBackend",
        *,
        device: Path,
        width: int,
        height: int,
        show_preview: bool = False,
        autostart: bool = True,
    ):
        self.backend = backend
        self.logger = logging.getLogger("app.vision.camera.usb_v4l2.stream")
        self.device = device
        self.width = width
        self.height = height
        self.show_preview = show_preview
        self.frame_index = 0
        self._started = False
        self._gst = self._load_gst()
        self._pipeline = self._build_pipeline()
        self._sink = self._pipeline.get_by_name("appsink0")
        self._preview_sink = self._pipeline.get_by_name("previewsink0")
        self._preview_widget = None
        if self._preview_sink is not None:
            try:
                self._preview_widget = self._preview_sink.get_property("widget")
            except Exception:
                self._preview_widget = None
        if self._sink is None:
            raise RuntimeError("USB GStreamer appsink 初始化失败")
        self._bus = self._pipeline.get_bus()
        self._overlay_boxes: list[dict] = []
        self._overlay_lock = threading.Lock()
        self._restart_lock = threading.Lock()
        self._cairo_overlay = self._pipeline.get_by_name("cairooverlay0")
        if self._cairo_overlay is not None:
            self._cairo_overlay.connect("draw", self._on_cairo_draw)
        if autostart:
            self.start()

    def start(self) -> None:
        if self._started:
            return
        state_change = self._pipeline.set_state(self._gst.State.PLAYING)
        self.logger.info("usb gst stream state change: %s", state_change.value_nick)
        self._started = True

    def capture_frame(self, output_path: Path | None = None) -> CapturedFrame:
        timeout_seconds = max(2, self.backend.config.usb.capture_timeout_seconds)
        sample = self._pull_sample_with_recovery(timeout_seconds=timeout_seconds)
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        pixel_format = str(structure.get_value("format"))

        buffer = sample.get_buffer()
        ok, info = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("USB GStreamer buffer 映射失败")

        try:
            stride = self._infer_stride(info.size, width, height)
            rgb = nv12_bytes_to_rgb(info.data, width=width, height=height, stride=stride)
        finally:
            buffer.unmap(info)

        output_path = output_path or self.backend._default_output_path(".jpg")
        self.frame_index += 1
        return CapturedFrame(
            backend=self.backend.backend_name,
            image_path=output_path,
            width=width,
            height=height,
            pixel_format=pixel_format,
            rgb=rgb,
            details={
                "device": str(self.device),
                "capture_mode": "gst_apps_sink",
                "stream_frame_index": self.frame_index,
                "stride": stride,
            },
        )

    def drain_frame(self, timeout_seconds: int = 1) -> None:
        sample = self._pull_sample_with_recovery(timeout_seconds=max(1, int(timeout_seconds)))
        del sample

    def restart_for_resume(self) -> None:
        self._restart_stream(reason="resume after preview-only pause")

    def set_overlay_boxes(self, boxes: list[dict]) -> None:
        with self._overlay_lock:
            self._overlay_boxes = list(boxes)

    def _on_cairo_draw(self, overlay, cr, timestamp, duration):
        with self._overlay_lock:
            boxes = list(self._overlay_boxes)
        if not boxes:
            return
        for box in boxes:
            x1 = int(box.get("x1", 0))
            y1 = int(box.get("y1", 0))
            x2 = int(box.get("x2", 0))
            y2 = int(box.get("y2", 0))
            if x2 <= x1 or y2 <= y1:
                continue
            cr.set_source_rgba(0.0, 1.0, 0.0, 0.85)
            cr.set_line_width(3.0)
            cr.rectangle(x1, y1, x2 - x1, y2 - y1)
            cr.stroke()
            label = str(box.get("label", ""))
            if label:
                try:
                    cr.set_font_size(16)
                    extents = cr.text_extents(label)
                    pad = 4
                    cr.set_source_rgba(0.0, 0.0, 0.0, 0.75)
                    cr.rectangle(x1, y1 - extents.height - pad * 2, extents.width + pad * 2, extents.height + pad * 2)
                    cr.fill()
                    cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)
                    cr.move_to(x1 + pad, y1 - pad)
                    cr.show_text(label)
                except Exception:
                    pass

    def close(self) -> None:
        self.detach_preview_widget()
        self._pipeline.set_state(self._gst.State.NULL)

    def _restart_stream(self, reason: str) -> None:
        with self._restart_lock:
            self.logger.warning("restart usb gst stream due to: %s", reason)
            last_error: Exception | None = None
            for restart_attempt in range(1, 4):
                try:
                    self._set_pipeline_null()
                    time.sleep(0.1 * restart_attempt)
                    refreshed_device = self._wait_for_available_device(
                        timeout_seconds=max(1.0, 0.8 * restart_attempt),
                    )
                    if refreshed_device != self.device:
                        self.logger.info(
                            "usb gst stream device refreshed: %s -> %s",
                            self.device,
                            refreshed_device,
                        )
                        self.device = refreshed_device
                    self._apply_source_device()
                    self._bus = self._pipeline.get_bus()
                    state_change = self._pipeline.set_state(self._gst.State.PLAYING)
                    self.logger.info(
                        "usb gst stream restart state change (attempt %s/3): %s",
                        restart_attempt,
                        state_change.value_nick,
                    )
                    if state_change == self._gst.StateChangeReturn.FAILURE:
                        raise RuntimeError(f"unable to reopen camera device {self.device}")
                    self._started = True
                    return
                except Exception as exc:
                    last_error = exc
                    self.logger.warning(
                        "usb gst stream restart attempt %s/3 failed: %s",
                        restart_attempt,
                        exc,
                    )
                    self._set_pipeline_null()
                    time.sleep(0.2 * restart_attempt)
            if last_error is not None:
                raise RuntimeError(f"USB GStreamer restart failed after retries: {last_error}") from last_error

    @staticmethod
    def _is_recoverable_stream_error(exc: RuntimeError) -> bool:
        message = str(exc)
        recoverable_markers = (
            "USB GStreamer 提前收到 EOS",
            "gst_v4l2_object_poll",
            "gst-resource-error-quark",
            "poll error 1",
            "无法从资源阅读",
        )
        return any(marker in message for marker in recoverable_markers)

    def _pull_sample_with_recovery(self, timeout_seconds: int):
        last_exc: Exception | None = None
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                return self._pull_sample(timeout_seconds=timeout_seconds)
            except RuntimeError as exc:
                last_exc = exc
                if attempt >= max_attempts - 1 or not self._is_recoverable_stream_error(exc):
                    raise
                self.logger.warning(
                    "usb gst stream recoverable failure (attempt %s/%s): %s",
                    attempt + 1,
                    max_attempts - 1,
                    exc,
                )
                self._restart_stream(reason=str(exc))
        assert last_exc is not None
        raise last_exc

    def _build_pipeline(self):
        if self.show_preview:
            pipeline = " ".join(
                [
                    "v4l2src",
                    "name=v4l2src0",
                    f"device={self.device}",
                    "!",
                    f"image/jpeg,width={self.width},height={self.height},framerate=30/1",
                    "!",
                    "jpegparse",
                    "!",
                    "spacemitdec",
                    "!",
                    "tee",
                    "name=t",
                    "t.",
                    "!",
                    "queue",
                    "leaky=downstream",
                    "max-size-buffers=2",
                    "!",
                    "video/x-raw,format=NV12",
                    "!",
                    "appsink",
                    "name=appsink0",
                    "emit-signals=false",
                    "sync=false",
                    "drop=true",
                    "max-buffers=1",
                    "wait-on-eos=false",
                    "t.",
                    "!",
                    "queue",
                    "leaky=downstream",
                    "max-size-buffers=2",
                    "!",
                    "video/x-raw,format=NV12",
                    "!",
                    "videoconvert",
                    "!",
                    "cairooverlay",
                    "name=cairooverlay0",
                    "!",
                    "gtkwaylandsink",
                    "name=previewsink0",
                    "sync=false",
                ]
            )
        else:
            pipeline = " ".join(
                [
                    "v4l2src",
                    "name=v4l2src0",
                    f"device={self.device}",
                    "!",
                    (
                        f"image/jpeg,width={self.width},height={self.height},"
                        "framerate=30/1"
                    ),
                    "!",
                    "typefind",
                    "!",
                    "spacemitdec",
                    "!",
                    "video/x-raw,format=NV12",
                    "!",
                    "appsink",
                    "name=appsink0",
                    "emit-signals=false",
                    "sync=false",
                    "drop=true",
                    "max-buffers=1",
                    "wait-on-eos=false",
                ]
            )
        return self._gst.parse_launch(pipeline)

    def _set_pipeline_null(self) -> None:
        try:
            self._pipeline.set_state(self._gst.State.NULL)
        except Exception:
            pass

    def _wait_for_available_device(self, timeout_seconds: float) -> Path:
        deadline = time.monotonic() + max(0.2, timeout_seconds)
        last_device = self.device
        while time.monotonic() < deadline:
            candidate = self.backend.resolve_stream_device(fallback_device=self.device)
            last_device = candidate
            if candidate.exists():
                return candidate
            time.sleep(0.1)
        return last_device

    def _apply_source_device(self) -> None:
        source = self._pipeline.get_by_name("v4l2src0")
        if source is None:
            return
        try:
            source.set_property("device", str(self.device))
        except Exception as exc:
            self.logger.warning("failed to refresh v4l2 source device %s: %s", self.device, exc)

    def _pull_sample(self, timeout_seconds: int):
        deadline = time.monotonic() + timeout_seconds
        last_warning = ""
        while time.monotonic() < deadline:
            sample = self._sink.emit("try-pull-sample", 200 * self._gst.MSECOND)
            if sample is not None:
                return sample
            last_warning = self._drain_bus_messages()
        detail = f"，最后一条总线消息: {last_warning}" if last_warning else ""
        raise RuntimeError(f"USB 连续取流超时，未收到新帧{detail}")

    def _drain_bus_messages(self) -> str:
        last_message = ""
        while True:
            message = self._bus.pop_filtered(
                self._gst.MessageType.ERROR | self._gst.MessageType.WARNING | self._gst.MessageType.EOS
            )
            if message is None:
                return last_message
            source_name = message.src.get_name() if message.src else "unknown"
            if message.type == self._gst.MessageType.ERROR:
                err, debug = message.parse_error()
                raise RuntimeError(f"USB GStreamer ERROR from {source_name}: {err}; debug={debug}")
            if message.type == self._gst.MessageType.EOS:
                raise RuntimeError(f"USB GStreamer 提前收到 EOS: {source_name}")
            warn, debug = message.parse_warning()
            last_message = f"{source_name}: {warn}; debug={debug}"
            self.logger.warning("usb gstreamer warning from %s: %s; debug=%s", source_name, warn, debug)

    @staticmethod
    def _infer_stride(buffer_size: int, width: int, height: int) -> int:
        stride = (buffer_size * 2) // (height * 3)
        return max(width, stride)

    @staticmethod
    def _load_gst():
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return Gst


def build_camera_backend(config: VisionConfig, backend_name: str | None = None) -> CameraBackend:
    selected = backend_name or config.backend
    if selected == "mipi_official":
        return OfficialMipiCameraBackend(config)
    if selected == "usb_v4l2":
        return UsbV4L2CameraBackend(config)
    raise ValueError(f"不支持的相机后端: {selected}")
