from __future__ import annotations

import time
from pathlib import Path

import gi
import numpy as np
from PIL import Image, ImageDraw, ImageFont

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk

from app.vision.types import DefectCandidate, VisionAnalysisResult


PROJECT_LABEL_COPY = {
    "corrosion": ("腐蚀疑似", "Corrosion"),
    "crevice_corrosion": ("缝隙腐蚀", "Crevice Corrosion"),
    "pitting_corrosion": ("点蚀", "Pitting Corrosion"),
    "uniform_corrosion": ("均匀腐蚀", "Uniform Corrosion"),
    "rust_like_corrosion": ("锈蚀疑似", "Rust-like Corrosion"),
    "coating_flaking_or_delamination": ("涂层剥落疑似", "Flaking / Delamination"),
    "chalking_or_powdering": ("粉化疑似", "Chalking / Powdering"),
    "cui_risk_visual_hint": ("保温层下腐蚀风险提示", "CUI Risk Hint"),
}


class CompetitionDisplay:
    WINDOW_NAME = "Phase 6 Multimodal Demo"

    def __init__(self, title: str, snapshot_path: Path | None = None):
        self.title = title
        self.snapshot_path = snapshot_path
        self._font_path = self._detect_font_path()
        self._window: Gtk.Window | None = None
        self._top_image: Gtk.Image | None = None
        self._video_image: Gtk.Image | None = None
        self._video_fixed = None
        self._video_widget = None
        self._right_image: Gtk.Image | None = None
        self._bottom_image: Gtk.Image | None = None
        self._window_ready = False
        self._should_close = False
        self._screen_size = self._detect_screen_size()
        self._latest_video_bytes: bytes = b""
        self._latest_video_glib = None
        self._latest_right_bytes: bytes = b""
        self._latest_right_glib = None
        self._latest_top_bytes: bytes = b""
        self._latest_top_glib = None
        self._latest_bottom_bytes: bytes = b""
        self._latest_bottom_glib = None
        self._last_result_signature: tuple | None = None
        self._last_bottom_signature: tuple | None = None
        self._layout = self._compute_layout(*self._screen_size)
        self._top_panel_pil: Image.Image | None = None
        self._video_panel_pil: Image.Image | None = None
        self._right_panel_pil: Image.Image | None = None
        self._bottom_panel_pil: Image.Image | None = None

    def attach_video_widget(self, widget) -> None:
        self._video_widget = widget

    def open(self) -> None:
        if self._window_ready:
            return
        screen = Gdk.Screen.get_default()
        visual = screen.get_rgba_visual()
        self._window = Gtk.Window(title=self.WINDOW_NAME)
        if visual is not None:
            self._window.set_visual(visual)
        self._window.connect("destroy", self._on_destroy)
        self._window.connect("key-press-event", self._on_key_press)
        self._window.set_decorated(False)
        self._window.fullscreen()
        self._window.maximize()
        self._window.stick()
        self._window.set_keep_above(True)
        self._window.set_app_paintable(True)

        provider = Gtk.CssProvider()
        provider.load_from_data(b"""
            window {
              background: transparent;
            }
            .transparent-bg {
              background: transparent;
            }
        """)
        style_ctx = self._window.get_style_context()
        style_ctx.add_provider(provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        Gtk.StyleContext.add_provider_for_screen(screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self._top_image = Gtk.Image()
        self._video_image = Gtk.Image()
        self._right_image = Gtk.Image()
        self._bottom_image = Gtk.Image()

        overlay = Gtk.Overlay()
        self._window.add(overlay)

        if self._video_widget is not None:
            parent = self._video_widget.get_parent()
            if parent is not None and hasattr(parent, "remove"):
                try:
                    parent.remove(self._video_widget)
                except Exception:
                    pass
            if parent is not None and hasattr(parent, "hide"):
                try:
                    parent.hide()
                except Exception:
                    pass
            vw, vh = self._layout["video_size"]
            self._video_widget.set_size_request(vw, vh)
            self._video_widget.set_halign(Gtk.Align.START)
            self._video_widget.set_valign(Gtk.Align.START)
            offset_y = self._layout["gap"] * 2 + self._layout["top_size"][1]
            self._video_widget.set_margin_start(self._layout["gap"])
            self._video_widget.set_margin_top(offset_y)
            overlay.add_overlay(self._video_widget)
            overlay.set_overlay_pass_through(self._video_widget, True)

        vw, vh = self._layout["video_size"]
        off_y = self._layout["gap"] * 2 + self._layout["top_size"][1]
        self._video_fixed = Gtk.Fixed()
        self._video_fixed.set_size_request(vw, vh)
        self._video_image.set_size_request(vw, vh)
        self._video_fixed.put(self._video_image, 0, 0)
        overlay.add_overlay(self._video_fixed)
        self._video_fixed.set_margin_start(self._layout["gap"])
        self._video_fixed.set_margin_top(off_y)
        overlay.set_overlay_pass_through(self._video_fixed, True)

        ui_root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=self._layout["gap"])
        ui_root.set_margin_top(self._layout["gap"])
        ui_root.set_margin_bottom(self._layout["gap"])
        ui_root.set_margin_start(self._layout["gap"])
        ui_root.set_margin_end(self._layout["gap"])

        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=self._layout["gap"])
        spacer = Gtk.Box()
        spacer_ctx = spacer.get_style_context()
        spacer_ctx.add_class("transparent-bg")
        content.pack_start(spacer, True, True, 0)
        content.pack_start(self._right_image, False, False, 0)

        ui_root.pack_start(self._top_image, False, False, 0)
        ui_root.pack_start(content, True, True, 0)
        ui_root.pack_start(self._bottom_image, False, False, 0)

        overlay.add_overlay(ui_root)

        self._window.show_all()
        self._set_video_fixed_visible(self._video_widget is None)
        self._render_static_bars()
        self._window_ready = True
        self._pump_events()
        self._last_top_bar_update_sec: int = -1
        self._video_panel_bg = None
        self._video_panel_bg_size = None
        self._last_snapshot_write: float = 0.0

    def _apply_dark_css(self, widget: Gtk.Widget) -> None:
        pass

    def close(self) -> None:
        if self._video_widget is not None:
            parent = self._video_widget.get_parent()
            if parent is not None and hasattr(parent, "remove"):
                try:
                    parent.remove(self._video_widget)
                except Exception:
                    pass
            if hasattr(self._video_widget, "hide"):
                try:
                    self._video_widget.hide()
                except Exception:
                    pass
        if self._window is not None:
            try:
                self._window.hide()
            except Exception:
                pass
            self._pump_events()
            self._window.destroy()
        self._window = None
        self._top_image = None
        self._video_image = None
        self._video_fixed = None
        self._video_widget = None
        self._right_image = None
        self._bottom_image = None
        self._window_ready = False

    def show(
        self,
        rgb: np.ndarray,
        *,
        result: VisionAnalysisResult | None,
        capture_seconds: float,
        analysis_seconds: float,
        loop_seconds: float,
        frame_index: int,
        source_frame_index: int,
        captured_frames: int,
        camera_backend: str,
        assistant_status: dict | None = None,
        native_video_hold: bool = False,
    ) -> int:
        self.open()
        t0 = time.perf_counter()
        if self._video_widget is not None:
            if native_video_hold:
                self._update_video_panel(rgb, result)
                self._set_video_fixed_visible(True)
            else:
                self._set_video_fixed_visible(False)
        else:
            self._update_video_panel(rgb, result)
            self._set_video_fixed_visible(True)
        current_sec = int(t0)
        if current_sec != self._last_top_bar_update_sec:
            self._last_top_bar_update_sec = current_sec
            self._update_phase6_top_bar(
                camera_backend,
                result.recognizer_backend if result else "spacemit_vision",
            )
        self._update_right_panel(
            result,
            capture_seconds,
            analysis_seconds,
            loop_seconds,
            frame_index,
            source_frame_index,
            captured_frames,
        )
        self._update_phase6_bottom_bar(assistant_status)
        if self._last_snapshot_write == 0.0 or t0 - self._last_snapshot_write > 5.0:
            self._last_snapshot_write = t0
            self._write_snapshot()
        self._pump_events()
        return 27 if self._should_close else -1

    def _set_video_fixed_visible(self, visible: bool) -> None:
        if self._video_fixed is None:
            return
        try:
            if visible:
                self._video_fixed.show()
            else:
                self._video_fixed.hide()
        except Exception:
            pass

    def _render_static_bars(self) -> None:
        self._update_phase6_bottom_bar(None)

    def _update_phase6_top_bar(self, camera_backend: str, recognizer_backend: str) -> None:
        w, h = self._layout["top_size"]
        image = Image.new("RGB", (w, h), (8, 26, 38))
        draw = ImageDraw.Draw(image, "RGBA")
        title_font = self._load_font(max(28, w // 60))
        section_font = self._load_font(max(18, w // 110))
        draw.rounded_rectangle(
            (0, 0, w - 1, h - 1),
            radius=28,
            fill=(8, 26, 38, 245),
            outline=(103, 200, 245, 100),
            width=2,
        )
        self._pill(draw, (20, 24, 112, 62), "LIVE", fill=(220, 62, 55, 240), font=section_font)
        draw.text(
            (136, 18),
            "Board-side Corrosion Expert Phase 6",
            font=title_font,
            fill=(244, 251, 255, 255),
        )
        draw.text(
            (136, 62),
            f"{camera_backend}  |  {recognizer_backend}  |  Offline Multimodal Demo",
            font=section_font,
            fill=(150, 186, 207, 255),
        )
        stamp = f"Muse Pi Pro  |  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        draw.text(
            (w - self._text_width(draw, stamp, section_font) - 24, 28),
            stamp,
            font=section_font,
            fill=(141, 240, 174, 255),
        )
        self._top_panel_pil = image.copy()
        self._set_image(self._top_image, image, slot="top")

    def _update_top_bar(self, camera_backend: str, recognizer_backend: str) -> None:
        w, h = self._layout["top_size"]
        image = Image.new("RGB", (w, h), (8, 26, 38))
        draw = ImageDraw.Draw(image, "RGBA")
        title_font = self._load_font(max(28, w // 60))
        section_font = self._load_font(max(18, w // 110))
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=28, fill=(8, 26, 38, 245), outline=(103, 200, 245, 100), width=2)
        self._pill(draw, (20, 24, 112, 62), "LIVE", fill=(220, 62, 55, 240), font=section_font)
        draw.text((136, 18), "板端防腐智能巡检 Phase 5", font=title_font, fill=(244, 251, 255, 255))
        draw.text((136, 62), f"{camera_backend}  |  {recognizer_backend}", font=section_font, fill=(150, 186, 207, 255))
        stamp = f"Muse Pi Pro  |  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        draw.text((w - self._text_width(draw, stamp, section_font) - 24, 28), stamp, font=section_font, fill=(141, 240, 174, 255))
        self._top_panel_pil = image.copy()
        self._set_image(self._top_image, image, slot="top")

    def _update_video_panel(self, rgb: np.ndarray, result: VisionAnalysisResult | None) -> None:
        w, h = self._layout["video_size"]
        bg_size = (w, h)
        if self._video_panel_bg is None or self._video_panel_bg_size != bg_size:
            self._video_panel_bg = Image.new("RGB", (w, h), (7, 18, 28))
            bg_draw = ImageDraw.Draw(self._video_panel_bg, "RGBA")
            bg_draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=28, fill=(7, 18, 28, 255), outline=(103, 200, 245, 82), width=2)
            inset = 18
            footer_h = 146
            footer_rect_cache = (inset, h - 128, w - inset, h - inset)
            bg_draw.rounded_rectangle(footer_rect_cache, radius=22, fill=(8, 24, 36, 242), outline=(103, 200, 245, 78), width=1)
            self._video_panel_bg_size = bg_size
        panel = self._video_panel_bg.copy()
        draw = ImageDraw.Draw(panel, "RGBA")

        inset = 18
        footer_h = 146
        frame_rect = (inset, inset, w - inset, h - footer_h)
        frame_img = self._fit_image_to_rect(rgb, frame_rect[2] - frame_rect[0], frame_rect[3] - frame_rect[1])
        panel.paste(frame_img, (frame_rect[0], frame_rect[1]))
        draw = ImageDraw.Draw(panel, "RGBA")
        draw.rounded_rectangle(frame_rect, radius=20, outline=(255, 255, 255, 24), width=1)
        self._draw_project_boxes(draw, frame_rect, result.candidates if result else [], self._load_font(max(18, w // 90)))

        footer_rect = (inset, h - 128, w - inset, h - inset)
        section_font = self._load_font(max(18, w // 100))
        hero_font = self._load_font(max(32, w // 42))
        body_font = self._load_font(max(18, w // 96))
        top_candidate = result.candidates[0] if (result and result.candidates) else None
        zh_label, en_label = self._display_label(top_candidate.label if top_candidate else "")
        summary = (top_candidate.summary if top_candidate else "正在稳定采集与分析当前画面").strip()
        score = top_candidate.score if top_candidate else 0.0
        draw.text((footer_rect[0] + 20, footer_rect[1] + 16), "主识别结果", font=section_font, fill=(92, 225, 230, 255))
        draw.text((footer_rect[0] + 20, footer_rect[1] + 44), zh_label or "实时分析中", font=hero_font, fill=(244, 251, 255, 255))
        if en_label:
            draw.text((footer_rect[0] + 22, footer_rect[1] + 86), en_label, font=section_font, fill=(156, 186, 200, 255))
        score_box = (footer_rect[2] - 154, footer_rect[1] + 20, footer_rect[2] - 18, footer_rect[1] + 78)
        draw.rounded_rectangle(score_box, radius=18, fill=(24, 42, 56, 255), outline=(255, 206, 104, 120), width=2)
        draw.text((score_box[0] + 16, score_box[1] + 10), "CONF", font=section_font, fill=(255, 206, 104, 255))
        draw.text((score_box[0] + 16, score_box[1] + 30), f"{score:.2f}", font=body_font, fill=(255, 246, 220, 255))
        self._wrapped_text(draw, footer_rect[0] + 300, footer_rect[1] + 22, footer_rect[2] - footer_rect[0] - 470, summary, body_font, (220, 232, 239, 255), 6, max_lines=3)
        self._video_panel_pil = panel.copy()
        self._set_image(self._video_image, panel, slot="video")

    def _update_right_panel(
        self,
        result: VisionAnalysisResult | None,
        capture_seconds: float,
        analysis_seconds: float,
        loop_seconds: float,
        frame_index: int,
        source_frame_index: int,
        captured_frames: int,
    ) -> None:
        sig_candidates = tuple(
            (c.label, round(c.score, 2)) for c in (result.candidates[:3] if result else [])
        )
        sig_timing = (
            int(source_frame_index / 3),
            round(capture_seconds, 1),
            round(analysis_seconds, 2),
            round(loop_seconds, 2),
        )
        signature = (sig_candidates, sig_timing)
        if signature == self._last_result_signature:
            return
        self._last_result_signature = signature

        w, h = self._layout["right_size"]
        panel = Image.new("RGB", (w, h), (8, 22, 32))
        draw = ImageDraw.Draw(panel, "RGBA")
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=28, fill=(8, 22, 32, 255), outline=(103, 200, 245, 90), width=2)
        title_font = self._load_font(max(26, w // 18))
        section_font = self._load_font(max(18, w // 26))
        body_font = self._load_font(max(18, w // 28))
        small_font = self._load_font(max(14, w // 34))
        metric_font = self._load_font(max(24, w // 20))
        draw.text((24, 20), "赛事状态面板", font=title_font, fill=(244, 251, 255, 255))
        draw.text((26, 64), "Competition Telemetry", font=section_font, fill=(152, 186, 204, 255))
        metric_y = 108
        metric_w = w - 48
        self._metric_card(draw, 24, metric_y, metric_w, "采集耗时", f"{capture_seconds:.3f}s", section_font, metric_font)
        self._metric_card(draw, 24, metric_y + 94, metric_w, "推理耗时", f"{analysis_seconds:.3f}s", section_font, metric_font)
        self._metric_card(draw, 24, metric_y + 188, metric_w, "单轮总耗时", f"{loop_seconds:.3f}s", section_font, metric_font)
        draw.text((24, metric_y + 292), "项目语义结果", font=section_font, fill=(92, 225, 230, 255))
        chip_y = metric_y + 324
        for candidate in (result.candidates[:3] if result else []):
            zh_label, _ = self._display_label(candidate.label)
            self._result_chip(draw, 24, chip_y, metric_w, zh_label or candidate.label, candidate.score, body_font, small_font)
            chip_y += 68
        if result is None or not result.candidates:
            self._result_chip(draw, 24, chip_y, metric_w, "当前无高置信结果", 0.0, body_font, small_font, muted=True)
            chip_y += 68
        draw.text((24, chip_y + 6), "运行信息", font=section_font, fill=(92, 225, 230, 255))
        info_y = chip_y + 40
        facts = [
            ("显示帧", str(frame_index)),
            ("分析源帧", str(source_frame_index)),
            ("累计采集", str(captured_frames)),
            ("候选结果", str(len(result.candidates) if result else 0)),
        ]
        for key, value in facts:
            draw.text((28, info_y), key, font=small_font, fill=(154, 186, 204, 255))
            draw.text((w - self._text_width(draw, value, body_font) - 28, info_y - 4), value, font=body_font, fill=(244, 251, 255, 255))
            info_y += 38
        self._right_panel_pil = panel.copy()
        self._set_image(self._right_image, panel, slot="right")

    def _update_bottom_bar(self) -> None:
        w, h = self._layout["bottom_size"]
        panel = Image.new("RGB", (w, h), (8, 22, 32))
        draw = ImageDraw.Draw(panel, "RGBA")
        body_font = self._load_font(max(18, w // 96))
        small_font = self._load_font(max(14, w // 120))
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=24, fill=(8, 22, 32, 255), outline=(103, 200, 245, 88), width=2)
        draw.text((20, 18), "SpaceMIT Vision Pipeline", font=body_font, fill=(244, 251, 255, 255))
        draw.text((290, 21), "USB Camera  |  Native Split-Screen OSD  |  ESC / Q 退出", font=small_font, fill=(160, 191, 207, 255))
        self._bottom_panel_pil = panel.copy()
        self._set_image(self._bottom_image, panel, slot="bottom")

    def _update_phase6_bottom_bar(self, assistant_status: dict | None = None) -> None:
        assistant_status = assistant_status or {}
        mode_text = self._clip_text(
            self._safe_status_text(assistant_status.get("mode"), "Inspection"),
            26,
        )
        stage_text = self._clip_text(
            self._safe_status_text(assistant_status.get("voice_stage"), "idle"),
            18,
        )
        headline_text = self._clip_text(
            self._safe_status_text(assistant_status.get("voice_headline"), "Voice and RAG ready"),
            36,
        )
        user_text = self._clip_text(
            self._safe_status_text(
                assistant_status.get("latest_user_text"),
                "Waiting for ASR or console question",
            ),
            52,
        )
        citation_items = assistant_status.get("latest_citations") or []
        citation_text = self._clip_text(
            ", ".join(str(item) for item in citation_items[:2]) or "No citations yet",
            48,
        )
        signature = (mode_text, stage_text, headline_text, user_text, citation_text)
        if signature == self._last_bottom_signature:
            return
        self._last_bottom_signature = signature

        w, h = self._layout["bottom_size"]
        panel = Image.new("RGB", (w, h), (8, 22, 32))
        draw = ImageDraw.Draw(panel, "RGBA")
        body_font = self._load_font(max(17, w // 100))
        small_font = self._load_font(max(13, w // 126))
        draw.rounded_rectangle(
            (0, 0, w - 1, h - 1),
            radius=24,
            fill=(8, 22, 32, 255),
            outline=(103, 200, 245, 88),
            width=2,
        )
        block_gap = 14
        block_y = 12
        block_h = h - 24
        left_w = max(280, int(w * 0.26))
        mid_w = max(300, int(w * 0.28))
        right_w = w - 40 - block_gap * 2 - left_w - mid_w
        left_x = 20
        mid_x = left_x + left_w + block_gap
        right_x = mid_x + mid_w + block_gap

        self._bottom_field(
            draw,
            left_x,
            block_y,
            left_w,
            block_h,
            "PIPELINE",
            "OFFLINE / USB camera / Native OSD / ESC or Q to exit",
            small_font,
            body_font,
        )
        self._bottom_field(
            draw,
            mid_x,
            block_y,
            mid_w,
            block_h,
            f"MODE {mode_text} / VOICE {stage_text}",
            headline_text,
            small_font,
            body_font,
        )
        self._bottom_field(
            draw,
            right_x,
            block_y,
            right_w,
            block_h,
            f"LATEST INPUT / {citation_text}",
            user_text,
            small_font,
            body_font,
        )
        self._bottom_panel_pil = panel.copy()
        self._set_image(self._bottom_image, panel, slot="bottom")

    def _write_snapshot(self) -> None:
        if self.snapshot_path is None:
            return
        if not all(
            panel is not None
            for panel in (
                self._top_panel_pil,
                self._video_panel_pil,
                self._right_panel_pil,
                self._bottom_panel_pil,
            )
        ):
            return
        gap = self._layout["gap"]
        screen_w, screen_h = self._screen_size
        canvas = Image.new("RGB", (screen_w, screen_h), (4, 16, 24))
        top = self._top_panel_pil
        video = self._video_panel_pil
        right = self._right_panel_pil
        bottom = self._bottom_panel_pil
        assert top is not None and video is not None and right is not None and bottom is not None
        canvas.paste(top, (gap, gap))
        content_y = gap * 2 + top.size[1]
        canvas.paste(video, (gap, content_y))
        canvas.paste(right, (gap * 2 + video.size[0], content_y))
        bottom_y = content_y + video.size[1] + gap
        canvas.paste(bottom, (gap, bottom_y))
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(self.snapshot_path)

    def _set_image(self, widget: Gtk.Image | None, image: Image.Image, slot: str) -> None:
        if widget is None:
            return
        rgb = np.asarray(image.convert("RGB"))
        data = np.ascontiguousarray(rgb).tobytes()
        glib_bytes = GLib.Bytes.new(data)
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            glib_bytes,
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            rgb.shape[1],
            rgb.shape[0],
            rgb.shape[1] * 3,
        )
        widget.set_from_pixbuf(pixbuf)
        if slot == "video":
            self._latest_video_bytes = data
            self._latest_video_glib = glib_bytes
        elif slot == "right":
            self._latest_right_bytes = data
            self._latest_right_glib = glib_bytes
        elif slot == "top":
            self._latest_top_bytes = data
            self._latest_top_glib = glib_bytes
        elif slot == "bottom":
            self._latest_bottom_bytes = data
            self._latest_bottom_glib = glib_bytes

    def _fit_image_to_rect(self, rgb: np.ndarray, target_w: int, target_h: int) -> Image.Image:
        image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        src_w, src_h = image.size
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        board = Image.new("RGB", (target_w, target_h), color=(10, 16, 22))
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        board.paste(resized, offset)
        return board

    def _draw_project_boxes(self, draw: ImageDraw.ImageDraw, frame_rect, candidates: list[DefectCandidate], body_font) -> None:
        fx1, fy1, fx2, fy2 = frame_rect
        frame_w = fx2 - fx1
        frame_h = fy2 - fy1
        src_w, src_h = 1280, 720
        scale = min(frame_w / src_w, frame_h / src_h)
        fitted_w = max(1, int(src_w * scale))
        fitted_h = max(1, int(src_h * scale))
        off_x = fx1 + (frame_w - fitted_w) // 2
        off_y = fy1 + (frame_h - fitted_h) // 2
        for candidate in candidates[:2]:
            box = candidate.box or {}
            if not box:
                continue
            x1 = off_x + int(float(box.get("x1", 0)) * scale)
            y1 = off_y + int(float(box.get("y1", 0)) * scale)
            x2 = off_x + int(float(box.get("x2", 0)) * scale)
            y2 = off_y + int(float(box.get("y2", 0)) * scale)
            x1 = max(fx1, min(fx2 - 1, x1))
            x2 = max(fx1, min(fx2 - 1, x2))
            y1 = max(fy1, min(fy2 - 1, y1))
            y2 = max(fy1, min(fy2 - 1, y2))
            if x2 <= x1 or y2 <= y1 or x2 - x1 > frame_w or y2 - y1 > frame_h:
                continue
            zh_label, _ = self._display_label(candidate.label)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=12, outline=(92, 225, 230, 255), width=4)
            label = f"{zh_label or candidate.label} {candidate.score:.2f}"
            tw = self._text_width(draw, label, body_font)
            tag_y = max(fy1 + 8, y1 - 38)
            draw.rounded_rectangle((x1, tag_y, x1 + tw + 24, tag_y + 32), radius=10, fill=(5, 22, 32, 240), outline=(92, 225, 230, 180), width=2)
            draw.text((x1 + 12, tag_y + 5), label, font=body_font, fill=(244, 251, 255, 255))

    def _metric_card(self, draw: ImageDraw.ImageDraw, x: int, y: int, width: int, label: str, value: str, small_font, big_font) -> None:
        draw.rounded_rectangle((x, y, x + width, y + 74), radius=22, fill=(12, 30, 42, 255), outline=(255, 255, 255, 18), width=1)
        draw.text((x + 18, y + 12), label, font=small_font, fill=(155, 188, 204, 255))
        draw.text((x + width - self._text_width(draw, value, big_font) - 18, y + 24), value, font=big_font, fill=(244, 251, 255, 255))

    def _result_chip(self, draw: ImageDraw.ImageDraw, x: int, y: int, width: int, label: str, score: float, body_font, small_font, muted: bool = False) -> None:
        fill = (12, 34, 46, 255) if not muted else (20, 28, 34, 255)
        outline = (120, 213, 255, 84) if not muted else (120, 120, 120, 50)
        draw.rounded_rectangle((x, y, x + width, y + 54), radius=18, fill=fill, outline=outline, width=1)
        draw.text((x + 16, y + 12), label, font=body_font, fill=(236, 246, 251, 255))
        score_text = f"{score:.2f}"
        draw.text((x + width - self._text_width(draw, score_text, body_font) - 18, y + 12), score_text, font=body_font, fill=(255, 206, 104, 255))
        draw.text((x + 16, y + 34), "project semantic output", font=small_font, fill=(140, 173, 189, 255))

    def _wrapped_text(self, draw: ImageDraw.ImageDraw, x: int, y: int, max_width: int, text: str, font, fill, line_gap: int, max_lines: int = 99) -> int:
        line = ""
        count = 0
        for ch in text:
            candidate = line + ch
            if self._text_width(draw, candidate, font) <= max_width:
                line = candidate
                continue
            draw.text((x, y), line, font=font, fill=fill)
            count += 1
            if count >= max_lines:
                return y
            y += self._line_height(font) + line_gap
            line = ch
        if line and count < max_lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += self._line_height(font)
        return y

    def _pump_events(self) -> None:
        deadline = time.perf_counter() + 0.012
        iterations = 0
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
            iterations += 1
            if iterations >= 128 or time.perf_counter() >= deadline:
                break

    def _on_destroy(self, *_args) -> None:
        self._should_close = True

    def _on_key_press(self, _widget, event) -> bool:
        if event.keyval in (Gdk.KEY_q, Gdk.KEY_Q, Gdk.KEY_Escape):
            self._should_close = True
            return True
        return False

    def _display_label(self, label: str) -> tuple[str, str]:
        return PROJECT_LABEL_COPY.get(label, (label.replace("_", " ").strip(), label.replace("_", " ").strip()))

    def _pill(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill, font) -> None:
        draw.rounded_rectangle(box, radius=16, fill=fill)
        tw = self._text_width(draw, text, font)
        th = self._line_height(font)
        draw.text((box[0] + ((box[2] - box[0]) - tw) / 2, box[1] + ((box[3] - box[1]) - th) / 2 - 2), text, font=font, fill=(255, 255, 255, 255))

    def _bottom_field(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        height: int,
        label: str,
        value: str,
        label_font,
        value_font,
    ) -> None:
        draw.rounded_rectangle(
            (x, y, x + width, y + height),
            radius=18,
            fill=(11, 31, 43, 255),
            outline=(255, 255, 255, 18),
            width=1,
        )
        draw.text((x + 14, y + 8), label, font=label_font, fill=(141, 240, 174, 255))
        self._wrapped_text(
            draw,
            x + 14,
            y + 26,
            width - 28,
            value,
            value_font,
            (236, 246, 251, 255),
            4,
            max_lines=2,
        )

    def _load_font(self, size: int):
        if self._font_path:
            try:
                return ImageFont.truetype(str(self._font_path), size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    @staticmethod
    def _safe_status_text(value, fallback: str) -> str:
        text = " ".join(str(value or "").replace("\n", " ").split()).strip()
        return text or fallback

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        text = " ".join(str(text).replace("\n", " ").split()).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _compute_layout(screen_w: int, screen_h: int) -> dict:
        gap = 24
        top_h = 112
        bottom_h = 84
        content_h = screen_h - gap * 4 - top_h - bottom_h
        right_w = max(430, int(screen_w * 0.26))
        left_w = screen_w - gap * 3 - right_w
        return {
            "gap": gap,
            "screen_size": (screen_w, screen_h),
            "top_size": (screen_w - gap * 2, top_h),
            "video_size": (left_w, content_h),
            "right_size": (right_w, content_h),
            "bottom_size": (screen_w - gap * 2, bottom_h),
        }

    @staticmethod
    def _detect_font_path() -> Path | None:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                return path
        return None

    @property
    def video_panel_size(self) -> tuple[int, int]:
        return self._layout["video_size"]

    @staticmethod
    def _detect_screen_size() -> tuple[int, int]:
        screen = Gdk.Screen.get_default()
        if screen is None:
            return (1920, 1080)
        return screen.get_width(), screen.get_height()

    @staticmethod
    def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    @staticmethod
    def _line_height(font) -> int:
        try:
            bbox = font.getbbox("Ag")
            return bbox[3] - bbox[1]
        except Exception:
            return max(16, getattr(font, "size", 16))
