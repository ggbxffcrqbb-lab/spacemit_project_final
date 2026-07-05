from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gtk


METRIC_BULLETS = [
    "推理性能: 图像 AI 推理帧率 / 本地大模型 Tokens per second / 首字延迟 TTFT",
    "交互体验: 语音与视觉识别准确率 / 任务切换连贯性 / 执行成功率",
    "资源管理: 模型运行时的 CPU NPU 内存占用可控",
    "系统稳定性: 长时间高负载运行无故障 / 数据传输稳定",
]


class SelectorDisplay:
    WINDOW_NAME = "Voice Guided Camera Selector"

    def __init__(self, title: str, snapshot_path: Path | None = None):
        self.title = title
        self.snapshot_path = snapshot_path
        self._window: Gtk.Window | None = None
        self._headline_label: Gtk.Label | None = None
        self._detail_label: Gtk.Label | None = None
        self._latest_label: Gtk.Label | None = None
        self._reply_label: Gtk.Label | None = None
        self._rag_label: Gtk.Label | None = None
        self._metrics_label: Gtk.Label | None = None
        self._health_label: Gtk.Label | None = None
        self._window_ready = False
        self._should_close = False

    def open(self) -> None:
        if self._window_ready:
            return
        screen = Gdk.Screen.get_default()
        self._window = Gtk.Window(title=self.WINDOW_NAME)
        self._window.connect("destroy", self._on_destroy)
        self._window.connect("key-press-event", self._on_key_press)
        self._window.set_decorated(False)
        self._window.fullscreen()
        self._window.maximize()
        self._window.stick()
        self._window.set_keep_above(True)

        provider = Gtk.CssProvider()
        provider.load_from_data(
            b"""
            window {
              background: #07111a;
            }
            .selector-root {
              background-image: linear-gradient(140deg, #07111a 0%, #0c2030 52%, #13263a 100%);
            }
            .selector-badge {
              color: #8be4b3;
              font-size: 16px;
              font-weight: 700;
            }
            .selector-hero {
              color: #ffffff;
              font-size: 52px;
              font-weight: 900;
            }
            .selector-sub {
              color: #c2d4df;
              font-size: 20px;
            }
            .selector-panel,
            .selector-card {
              background: rgba(11, 24, 37, 0.92);
              border-radius: 24px;
              border: 1px solid rgba(124, 197, 228, 0.18);
            }
            .selector-card-usb {
              background: rgba(10, 38, 55, 0.95);
              border-radius: 24px;
              border: 2px solid rgba(90, 196, 232, 0.42);
            }
            .selector-card-mipi {
              background: rgba(39, 31, 23, 0.95);
              border-radius: 24px;
              border: 2px solid rgba(232, 188, 121, 0.46);
            }
            .selector-card-title {
              color: #f6fbff;
              font-size: 26px;
              font-weight: 800;
            }
            .selector-card-subtitle {
              color: #86d8ea;
              font-size: 17px;
              font-weight: 700;
            }
            .selector-card-subtitle-warm {
              color: #e8bc79;
              font-size: 17px;
              font-weight: 700;
            }
            .selector-body {
              color: #c8d7e1;
              font-size: 17px;
            }
            .selector-section {
              color: #7fd7ea;
              font-size: 16px;
              font-weight: 700;
            }
            .selector-strong {
              color: #ffffff;
              font-size: 20px;
              font-weight: 800;
            }
            .selector-muted {
              color: #96acbb;
              font-size: 15px;
            }
        """
        )
        if screen is not None:
            Gtk.StyleContext.add_provider_for_screen(screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=22)
        root.set_margin_top(32)
        root.set_margin_bottom(28)
        root.set_margin_start(36)
        root.set_margin_end(36)
        root.get_style_context().add_class("selector-root")
        self._window.add(root)

        title_block = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        badge = Gtk.Label(label="BOARD-SIDE MULTIMODAL DEMO")
        badge.set_xalign(0.0)
        badge.get_style_context().add_class("selector-badge")
        hero = Gtk.Label(label="语音引导智能巡检总控台")
        hero.set_xalign(0.0)
        hero.get_style_context().add_class("selector-hero")
        sub = Gtk.Label(
            label="请先说“你好”，再选择智能巡检相机。当前页面面向竞赛答辩展示，突出推理性能、交互体验、资源管理与系统稳定性。"
        )
        sub.set_xalign(0.0)
        sub.set_line_wrap(True)
        sub.get_style_context().add_class("selector-sub")
        title_block.pack_start(badge, False, False, 0)
        title_block.pack_start(hero, False, False, 0)
        title_block.pack_start(sub, False, False, 0)

        center = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=22)

        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        left.set_hexpand(True)
        cards = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        cards.set_homogeneous(True)
        cards.pack_start(
            self._build_card(
                css_name="selector-card-usb",
                title="进入相机一智能巡检",
                subtitle="USB Camera / 大场景巡检",
                description="强调原生预览的流畅性、实时叠框反馈，以及更大视野下的智能巡检效率。",
                warm=False,
            ),
            True,
            True,
            0,
        )
        cards.pack_start(
            self._build_card(
                css_name="selector-card-mipi",
                title="进入相机二智能巡检",
                subtitle="MIPI Camera / 近距离细节分析",
                description="强调近距离细节观察、双阶段缺陷识别，以及局部风险精读能力。",
                warm=True,
            ),
            True,
            True,
            0,
        )

        indicator_panel = self._build_panel("参考重要考核指标")
        indicator_body = self._panel_body(indicator_panel)
        for bullet in METRIC_BULLETS:
            row = Gtk.Label(label=f"• {bullet}")
            row.set_xalign(0.0)
            row.set_line_wrap(True)
            row.get_style_context().add_class("selector-body")
            indicator_body.pack_start(row, False, False, 0)
        left.pack_start(cards, True, True, 0)
        left.pack_start(indicator_panel, False, False, 0)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        right.set_size_request(470, -1)

        status_panel = self._build_panel("当前运行状态")
        status_body = self._panel_body(status_panel)
        self._headline_label = Gtk.Label(label="等待唤醒词")
        self._headline_label.set_xalign(0.0)
        self._headline_label.set_line_wrap(True)
        self._headline_label.get_style_context().add_class("selector-strong")
        self._detail_label = Gtk.Label(label="请先说“你好”，再选择智能巡检相机")
        self._detail_label.set_xalign(0.0)
        self._detail_label.set_line_wrap(True)
        self._detail_label.get_style_context().add_class("selector-body")
        self._metrics_label = Gtk.Label(label="TTFT / Tokens per second / 命中条数 / 视觉指标将在运行后实时显示")
        self._metrics_label.set_xalign(0.0)
        self._metrics_label.set_line_wrap(True)
        self._metrics_label.get_style_context().add_class("selector-muted")
        self._health_label = Gtk.Label(label="知识库文档数、索引块数，以及当前模式将在此处显示")
        self._health_label.set_xalign(0.0)
        self._health_label.set_line_wrap(True)
        self._health_label.get_style_context().add_class("selector-muted")
        status_body.pack_start(self._headline_label, False, False, 0)
        status_body.pack_start(self._detail_label, False, False, 0)
        status_body.pack_start(self._metrics_label, False, False, 0)
        status_body.pack_start(self._health_label, False, False, 0)

        qa_panel = self._build_panel("最近一次问答与知识命中")
        qa_body = self._panel_body(qa_panel)
        self._latest_label = Gtk.Label(label="最近输入: 暂无")
        self._latest_label.set_xalign(0.0)
        self._latest_label.set_line_wrap(True)
        self._latest_label.get_style_context().add_class("selector-body")
        self._reply_label = Gtk.Label(label="最近回答: 暂无")
        self._reply_label.set_xalign(0.0)
        self._reply_label.set_line_wrap(True)
        self._reply_label.get_style_context().add_class("selector-body")
        self._rag_label = Gtk.Label(label="知识库命中: 暂无")
        self._rag_label.set_xalign(0.0)
        self._rag_label.set_line_wrap(True)
        self._rag_label.get_style_context().add_class("selector-muted")
        qa_body.pack_start(self._latest_label, False, False, 0)
        qa_body.pack_start(self._reply_label, False, False, 0)
        qa_body.pack_start(self._rag_label, False, False, 0)

        right.pack_start(status_panel, False, False, 0)
        right.pack_start(qa_panel, True, True, 0)

        center.pack_start(left, True, True, 0)
        center.pack_start(right, False, False, 0)

        root.pack_start(title_block, False, False, 0)
        root.pack_start(center, True, True, 0)

        self._window.show_all()
        self._window_ready = True
        self._should_close = False
        self._pump_events()

    def show(self, assistant_status: dict | None = None) -> int:
        self.open()
        assistant_status = assistant_status or {}
        headline = self._safe_text(assistant_status.get("voice_headline"), "等待唤醒词")
        detail = self._safe_text(assistant_status.get("voice_detail"), "请先说“你好”，再选择智能巡检相机")
        latest_user = self._safe_text(assistant_status.get("latest_user_text"), "暂无")
        latest_reply = self._safe_text(assistant_status.get("latest_reply_text"), "暂无")
        rag_hits = list(assistant_status.get("latest_rag_hits") or [])
        voice_metrics = dict(assistant_status.get("latest_voice_metrics") or {})
        latest_visual_summary = self._safe_text(assistant_status.get("latest_visual_summary"), "视觉结果待更新")
        mode = self._safe_text(assistant_status.get("mode"), "Camera Select")
        rag_docs = int(assistant_status.get("rag_document_count", 0) or 0)
        rag_chunks = int(assistant_status.get("rag_chunk_count", 0) or 0)

        rag_text = "知识库命中: 暂无"
        if rag_hits:
            lines = []
            for hit in rag_hits[:2]:
                title = self._safe_text(hit.get("title"), "未命名资料")
                score = hit.get("score", "--")
                lines.append(f"{title} (score={score})")
            rag_text = "知识库命中: " + " | ".join(lines)

        metric_parts = []
        if "first_chunk_ms" in voice_metrics:
            metric_parts.append(f"TTFT {voice_metrics.get('first_chunk_ms')} ms")
        if "total_ms" in voice_metrics:
            metric_parts.append(f"总耗时 {voice_metrics.get('total_ms')} ms")
        if "retrieved_hits" in voice_metrics:
            metric_parts.append(f"命中 {voice_metrics.get('retrieved_hits')} 条")
        metrics_text = " | ".join(metric_parts) if metric_parts else "TTFT / Tokens per second / 命中条数 / 视觉指标将在运行后实时显示"
        health_text = f"当前模式: {mode} | 知识库文档 {rag_docs} 份 | 索引块 {rag_chunks} 个 | 视觉摘要: {latest_visual_summary}"

        self._set_text(self._headline_label, headline)
        self._set_text(self._detail_label, detail)
        self._set_text(self._latest_label, f"最近输入: {latest_user}")
        self._set_text(self._reply_label, f"最近回答: {latest_reply}")
        self._set_text(self._rag_label, rag_text)
        self._set_text(self._metrics_label, metrics_text)
        self._set_text(self._health_label, health_text)
        self._pump_events()
        return 27 if self._should_close else -1

    def close(self) -> None:
        if self._window is not None:
            try:
                self._window.hide()
            except Exception:
                pass
            self._pump_events()
            self._window.destroy()
        self._window = None
        self._headline_label = None
        self._detail_label = None
        self._latest_label = None
        self._reply_label = None
        self._rag_label = None
        self._metrics_label = None
        self._health_label = None
        self._window_ready = False
        self._should_close = False

    def _build_card(self, *, css_name: str, title: str, subtitle: str, description: str, warm: bool) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_hexpand(True)
        box.set_vexpand(True)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(18)
        box.set_margin_end(18)
        box.get_style_context().add_class(css_name)

        title_label = Gtk.Label(label=title)
        title_label.set_xalign(0.0)
        title_label.set_line_wrap(True)
        title_label.get_style_context().add_class("selector-card-title")

        subtitle_label = Gtk.Label(label=subtitle)
        subtitle_label.set_xalign(0.0)
        subtitle_label.get_style_context().add_class("selector-card-subtitle-warm" if warm else "selector-card-subtitle")

        desc_label = Gtk.Label(label=description)
        desc_label.set_xalign(0.0)
        desc_label.set_line_wrap(True)
        desc_label.get_style_context().add_class("selector-body")

        box.pack_start(title_label, False, False, 0)
        box.pack_start(subtitle_label, False, False, 0)
        box.pack_start(desc_label, False, False, 0)
        return box

    def _build_panel(self, title: str) -> Gtk.Box:
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        panel.get_style_context().add_class("selector-panel")
        head = Gtk.Label(label=title)
        head.set_xalign(0.0)
        head.set_margin_top(16)
        head.set_margin_start(18)
        head.set_margin_end(18)
        head.get_style_context().add_class("selector-section")
        panel.pack_start(head, False, False, 0)
        return panel

    def _panel_body(self, panel: Gtk.Box) -> Gtk.Box:
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        body.set_margin_top(12)
        body.set_margin_bottom(18)
        body.set_margin_start(18)
        body.set_margin_end(18)
        panel.pack_start(body, True, True, 0)
        return body

    def _pump_events(self) -> None:
        iterations = 0
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
            iterations += 1
            if iterations >= 64:
                break

    def _on_destroy(self, *_args) -> None:
        self._should_close = True

    def _on_key_press(self, _widget, event) -> bool:
        if event.keyval in (Gdk.KEY_q, Gdk.KEY_Q, Gdk.KEY_Escape):
            self._should_close = True
            return True
        return False

    @staticmethod
    def _safe_text(value, fallback: str) -> str:
        text = " ".join(str(value or "").replace("\n", " ").split()).strip()
        return text or fallback

    @staticmethod
    def _set_text(label: Gtk.Label | None, text: str) -> None:
        if label is not None:
            label.set_text(text)
