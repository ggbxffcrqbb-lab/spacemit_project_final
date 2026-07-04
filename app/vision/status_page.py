from __future__ import annotations

from html import escape
import json
import threading
import time
from pathlib import Path
from typing import Any


class VisionStatusPageWriter:
    def __init__(self, output_dir: Path, title: str, refresh_seconds: int):
        self.output_dir = output_dir
        self.title = title
        self.refresh_seconds = max(1, int(refresh_seconds))
        self.html_path = self.output_dir / "vision_status.html"
        self.json_path = self.output_dir / "vision_status.json"
        self.text_path = self.output_dir / "vision_status.txt"
        self._lock = threading.RLock()
        self._html_written = False
        self._state: dict[str, Any] = {
            "title": title,
            "stage": "starting",
            "headline": "等待视觉流启动",
            "detail": "",
            "camera_backend": "",
            "recognizer_backend": "",
            "frame_index": 0,
            "frame_interval_seconds": 0.0,
            "capture_seconds": 0.0,
            "analysis_seconds": 0.0,
            "loop_seconds": 0.0,
            "latest_capture_path": "",
            "latest_annotated_path": "",
            "latest_candidates": [],
            "latest_metrics": {},
            "latest_notes": [],
            "updated_at": self._now(),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._render_locked()

    def update(
        self,
        *,
        stage: str,
        headline: str,
        detail: str = "",
        camera_backend: str | None = None,
        recognizer_backend: str | None = None,
        frame_index: int | None = None,
        frame_interval_seconds: float | None = None,
        capture_seconds: float | None = None,
        analysis_seconds: float | None = None,
        loop_seconds: float | None = None,
        latest_capture_path: str | None = None,
        latest_annotated_path: str | None = None,
        latest_candidates: list[dict[str, Any]] | None = None,
        latest_metrics: dict[str, Any] | None = None,
        latest_notes: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._state["stage"] = stage
            self._state["headline"] = headline
            self._state["detail"] = detail
            self._state["updated_at"] = self._now()
            if camera_backend is not None:
                self._state["camera_backend"] = camera_backend
            if recognizer_backend is not None:
                self._state["recognizer_backend"] = recognizer_backend
            if frame_index is not None:
                self._state["frame_index"] = frame_index
            if frame_interval_seconds is not None:
                self._state["frame_interval_seconds"] = round(frame_interval_seconds, 3)
            if capture_seconds is not None:
                self._state["capture_seconds"] = round(capture_seconds, 3)
            if analysis_seconds is not None:
                self._state["analysis_seconds"] = round(analysis_seconds, 3)
            if loop_seconds is not None:
                self._state["loop_seconds"] = round(loop_seconds, 3)
            if latest_capture_path is not None:
                self._state["latest_capture_path"] = latest_capture_path
            if latest_annotated_path is not None:
                self._state["latest_annotated_path"] = latest_annotated_path
            if latest_candidates is not None:
                self._state["latest_candidates"] = latest_candidates
            if latest_metrics is not None:
                self._state["latest_metrics"] = latest_metrics
            if latest_notes is not None:
                self._state["latest_notes"] = latest_notes
            self._render_locked()

    def _render_locked(self) -> None:
        self._write_text_atomic(
            self.json_path,
            json.dumps(self._state, ensure_ascii=False, separators=(",", ":")) + "\n",
        )
        self._write_text_atomic(self.text_path, self._render_text())
        if not self._html_written:
            self._write_text_atomic(self.html_path, self._render_html())
            self._html_written = True

    def _render_text(self) -> str:
        state = self._state
        lines = [
            self.title,
            "=" * len(self.title),
            f"阶段: {state['stage']}",
            f"状态: {state['headline']}",
            f"说明: {state['detail']}",
            f"相机后端: {state['camera_backend'] or '未知'}",
            f"识别后端: {state['recognizer_backend'] or '未知'}",
            f"处理帧序号: {state['frame_index']}",
            f"目标间隔: {state['frame_interval_seconds']} s",
            f"采集耗时: {state['capture_seconds']} s",
            f"推理耗时: {state['analysis_seconds']} s",
            f"单轮耗时: {state['loop_seconds']} s",
            f"画面文件: {state['latest_capture_path'] or '暂无'}",
            f"更新时间: {state['updated_at']}",
            "",
            "候选结果:",
        ]
        candidates = state.get("latest_candidates", [])
        if candidates:
            for index, candidate in enumerate(candidates, start=1):
                label = candidate.get("label", "unknown")
                score = candidate.get("score", 0)
                lines.append(f"{index}. {label} | score={score}")
                summary = candidate.get("summary", "")
                if summary:
                    lines.append(f"   {summary}")
        else:
            lines.append("暂无")
        return "\n".join(lines) + "\n"

    def _render_html(self) -> str:
        bootstrap = escape(json.dumps(self._state, ensure_ascii=False))
        page_title = escape(self.title)
        output_dir_json = json.dumps(str(self.output_dir))
        poll_interval_ms = max(300, min(900, self.refresh_seconds * 400))
        json_name = escape(self.json_path.name)

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
    :root {{
      --bg-0: #031018;
      --bg-1: #0a1f2c;
      --panel: rgba(7, 24, 35, 0.9);
      --panel-strong: rgba(10, 31, 44, 0.96);
      --line: rgba(121, 198, 255, 0.18);
      --ink: #f4fbff;
      --muted: #8fb0c3;
      --accent: #5ce1e6;
      --accent-2: #8df0ae;
      --warn: #ffcb66;
      --danger: #ff7b72;
      --shadow: 0 22px 60px rgba(0, 0, 0, 0.35);
    }}
    * {{
      box-sizing: border-box;
    }}
    html, body {{
      width: 100%;
      height: 100%;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Source Han Sans SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at 14% 18%, rgba(92, 225, 230, 0.18), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(255, 203, 102, 0.12), transparent 18%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
      overflow: hidden;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
      background-size: 28px 28px;
      mask-image: linear-gradient(180deg, rgba(255, 255, 255, 0.55), transparent 92%);
      pointer-events: none;
    }}
    .page {{
      position: relative;
      z-index: 1;
      width: 100vw;
      height: 100vh;
      padding: 18px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .topbar {{
      padding: 18px 22px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .title {{
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
      font-weight: 800;
    }}
    .subtitle {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--ink);
      font-size: 13px;
      white-space: nowrap;
    }}
    .badge-live::before {{
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #ff5147;
      box-shadow: 0 0 0 rgba(255, 81, 71, 0.35);
      animation: pulse 1.6s infinite;
    }}
    .content {{
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1.75fr) minmax(380px, 0.95fr);
      gap: 16px;
    }}
    .stage-panel {{
      min-height: 0;
      padding: 14px;
    }}
    .camera-shell {{
      position: relative;
      height: 100%;
      overflow: hidden;
      border-radius: 18px;
      background:
        radial-gradient(circle at 18% 18%, rgba(92, 225, 230, 0.2), transparent 22%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01)),
        #04111a;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .camera-shell img {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center center;
      display: none;
      transform: scale(1.01);
    }}
    .camera-overlay {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(180deg, rgba(2, 9, 15, 0.14), rgba(2, 9, 15, 0.04) 24%, rgba(2, 9, 15, 0.7) 100%);
    }}
    .camera-top,
    .camera-bottom {{
      position: absolute;
      left: 18px;
      right: 18px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      z-index: 1;
    }}
    .camera-top {{
      top: 18px;
      align-items: flex-start;
    }}
    .camera-bottom {{
      bottom: 18px;
      align-items: end;
    }}
    .camera-copy {{
      max-width: min(64%, 720px);
    }}
    .stage-tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 14px;
      border-radius: 999px;
      background: rgba(5, 18, 27, 0.62);
      border: 1px solid rgba(92, 225, 230, 0.28);
      color: var(--accent);
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .headline {{
      margin: 0;
      font-size: 34px;
      line-height: 1.08;
      font-weight: 800;
      text-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
    }}
    .detail {{
      margin: 12px 0 0;
      color: rgba(244, 251, 255, 0.84);
      font-size: 15px;
      line-height: 1.6;
      white-space: pre-wrap;
      text-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
    }}
    .quick-metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(140px, 180px));
      gap: 10px;
    }}
    .quick-metric {{
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(5, 18, 27, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.08);
      text-align: right;
    }}
    .quick-metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .quick-metric strong {{
      display: block;
      margin-top: 6px;
      font-size: 24px;
      font-weight: 800;
      color: var(--ink);
    }}
    .side {{
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto auto 1fr;
      gap: 16px;
    }}
    .card {{
      min-height: 0;
      padding: 18px;
    }}
    .card-title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .card-title {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .card-meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .result-hero {{
      background:
        radial-gradient(circle at top left, rgba(92, 225, 230, 0.16), transparent 24%),
        linear-gradient(180deg, rgba(10, 31, 44, 0.98), rgba(7, 24, 35, 0.96));
    }}
    .top-label {{
      font-size: 30px;
      line-height: 1.12;
      font-weight: 800;
      margin: 8px 0 10px;
    }}
    .top-summary {{
      color: var(--muted);
      line-height: 1.65;
      min-height: 52px;
    }}
    .confidence-row {{
      margin-top: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .confidence-bar {{
      margin-top: 10px;
      width: 100%;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.08);
    }}
    .confidence-bar > div {{
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--warn));
      transition: width 180ms ease-out;
    }}
    .candidate-list,
    .notes-list {{
      display: grid;
      gap: 10px;
    }}
    .candidate-item {{
      padding: 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.07);
    }}
    .candidate-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    .candidate-name {{
      font-size: 18px;
      font-weight: 700;
      line-height: 1.3;
    }}
    .candidate-score {{
      flex: 0 0 auto;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255, 203, 102, 0.14);
      border: 1px solid rgba(255, 203, 102, 0.28);
      color: var(--warn);
      font-size: 13px;
      font-weight: 700;
    }}
    .candidate-summary {{
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric-chip {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.07);
      min-width: 0;
    }}
    .metric-key {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-value {{
      font-size: 14px;
      line-height: 1.4;
      font-weight: 700;
      word-break: break-word;
    }}
    .note-pill {{
      display: inline-flex;
      align-items: center;
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(92, 225, 230, 0.08);
      border: 1px solid rgba(92, 225, 230, 0.14);
      color: rgba(244, 251, 255, 0.86);
      line-height: 1.5;
      font-size: 14px;
    }}
    .empty-state {{
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      font-size: 18px;
      line-height: 1.7;
    }}
    .muted {{
      color: var(--muted);
    }}
    @keyframes pulse {{
      0% {{
        box-shadow: 0 0 0 0 rgba(255, 81, 71, 0.4);
      }}
      70% {{
        box-shadow: 0 0 0 10px rgba(255, 81, 71, 0.0);
      }}
      100% {{
        box-shadow: 0 0 0 0 rgba(255, 81, 71, 0.0);
      }}
    }}
    @media (max-width: 1280px) {{
      .content {{
        grid-template-columns: 1fr;
      }}
      .side {{
        grid-template-rows: repeat(4, auto);
      }}
      .camera-bottom {{
        flex-direction: column;
        align-items: stretch;
      }}
      .camera-copy {{
        max-width: 100%;
      }}
      .quick-metrics {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 900px) {{
      body {{
        overflow: auto;
      }}
      .page {{
        height: auto;
        min-height: 100vh;
      }}
      .topbar {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .badge-row {{
        justify-content: flex-start;
      }}
      .headline {{
        font-size: 26px;
      }}
      .quick-metrics,
      .metric-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <script id="bootstrap-data" type="application/json">{bootstrap}</script>
  <div class="page">
    <header class="topbar panel">
      <div>
        <div class="eyebrow">Muse Pi Pro Phase 5</div>
        <h1 id="page-title" class="title">{page_title}</h1>
        <p class="subtitle">USB 实时采集 · SpaceMIT 官方视觉后端 · 比赛演示模式</p>
      </div>
      <div class="badge-row">
        <span class="badge badge-live">LIVE</span>
        <span id="camera-badge" class="badge">相机后端待连接</span>
        <span id="recognizer-badge" class="badge">识别后端待连接</span>
      </div>
    </header>

    <main class="content">
      <section class="panel stage-panel">
        <div class="camera-shell">
          <img id="main-image" alt="vision stream">
          <div id="empty-image" class="empty-state">正在等待板端视频流与识别结果…</div>
          <div class="camera-overlay"></div>
          <div class="camera-top">
            <div class="badge badge-live">实时演示</div>
            <div id="updated-at" class="badge">更新时间 --</div>
          </div>
          <div class="camera-bottom">
            <div class="camera-copy">
              <div id="stage" class="stage-tag">starting</div>
              <h2 id="headline" class="headline">等待视觉流启动</h2>
              <p id="detail" class="detail">系统将自动突出当前最值得评委关注的目标。</p>
            </div>
            <div class="quick-metrics">
              <div class="quick-metric">
                <span>Frame</span>
                <strong id="frame-index">0</strong>
              </div>
              <div class="quick-metric">
                <span>Capture</span>
                <strong id="capture-seconds">0.000s</strong>
              </div>
              <div class="quick-metric">
                <span>Infer</span>
                <strong id="analysis-seconds">0.000s</strong>
              </div>
              <div class="quick-metric">
                <span>Loop</span>
                <strong id="loop-seconds">0.000s</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      <aside class="side">
        <section class="panel card result-hero">
          <div class="eyebrow">Judging Focus</div>
          <div id="top-label" class="top-label">等待识别结果</div>
          <div id="top-summary" class="top-summary">这里会突出当前画面里最重要的缺陷语义，方便评委快速理解系统判断。</div>
          <div class="confidence-row">
            <span>Confidence</span>
            <span id="top-score">0.00</span>
          </div>
          <div class="confidence-bar">
            <div id="top-score-bar"></div>
          </div>
        </section>

        <section class="panel card">
          <div class="card-title-row">
            <h3 class="card-title">候选结果</h3>
            <span id="candidate-count" class="card-meta">0 项</span>
          </div>
          <div id="candidate-list" class="candidate-list"></div>
        </section>

        <section class="panel card">
          <div class="card-title-row">
            <h3 class="card-title">运行指标</h3>
            <span class="card-meta">实时更新</span>
          </div>
          <div id="metric-grid" class="metric-grid"></div>
        </section>

        <section class="panel card">
          <div class="card-title-row">
            <h3 class="card-title">补充说明</h3>
            <span class="card-meta">答辩可直接口播</span>
          </div>
          <div id="notes-list" class="notes-list"></div>
        </section>
      </aside>
    </main>
  </div>

  <script>
    const OUTPUT_DIR = {output_dir_json};
    const PAGE_TITLE = {json.dumps(self.title)};
    const JSON_URL = "./{json_name}";
    const POLL_INTERVAL_MS = {poll_interval_ms};
    const bootstrap = JSON.parse(document.getElementById("bootstrap-data").textContent);

    const refs = {{
      pageTitle: document.getElementById("page-title"),
      cameraBadge: document.getElementById("camera-badge"),
      recognizerBadge: document.getElementById("recognizer-badge"),
      updatedAt: document.getElementById("updated-at"),
      stage: document.getElementById("stage"),
      headline: document.getElementById("headline"),
      detail: document.getElementById("detail"),
      frameIndex: document.getElementById("frame-index"),
      captureSeconds: document.getElementById("capture-seconds"),
      analysisSeconds: document.getElementById("analysis-seconds"),
      loopSeconds: document.getElementById("loop-seconds"),
      mainImage: document.getElementById("main-image"),
      emptyImage: document.getElementById("empty-image"),
      topLabel: document.getElementById("top-label"),
      topSummary: document.getElementById("top-summary"),
      topScore: document.getElementById("top-score"),
      topScoreBar: document.getElementById("top-score-bar"),
      candidateCount: document.getElementById("candidate-count"),
      candidateList: document.getElementById("candidate-list"),
      metricGrid: document.getElementById("metric-grid"),
      notesList: document.getElementById("notes-list"),
    }};

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function formatSeconds(value) {{
      const number = Number(value ?? 0);
      if (!Number.isFinite(number)) {{
        return "0.000s";
      }}
      return `${{number.toFixed(3)}}s`;
    }}

    function formatScore(value) {{
      const number = Number(value ?? 0);
      if (!Number.isFinite(number)) {{
        return "0.00";
      }}
      return number.toFixed(2);
    }}

    function assetUrl(path) {{
      if (!path) {{
        return "";
      }}
      const normalized = String(path).replace(/\\\\/g, "/");
      const root = String(OUTPUT_DIR).replace(/\\\\/g, "/");
      if (normalized === root) {{
        return "";
      }}
      if (normalized.startsWith(root + "/")) {{
        return encodeURI(normalized.slice(root.length + 1));
      }}
      return normalized;
    }}

    function renderTopCandidate(candidates) {{
      const top = Array.isArray(candidates) && candidates.length > 0 ? candidates[0] : null;
      if (!top) {{
        refs.topLabel.textContent = "等待识别结果";
        refs.topSummary.textContent = "这里会突出当前画面里最重要的缺陷语义，方便评委快速理解系统判断。";
        refs.topScore.textContent = "0.00";
        refs.topScoreBar.style.width = "0%";
        return;
      }}
      const score = Math.max(0, Math.min(1, Number(top.score || 0)));
      refs.topLabel.textContent = top.label || "unknown";
      refs.topSummary.textContent = top.summary || "当前候选结果已进入展示区，可直接结合左侧实时画面进行讲解。";
      refs.topScore.textContent = formatScore(score);
      refs.topScoreBar.style.width = `${{Math.round(score * 100)}}%`;
    }}

    function renderCandidates(candidates) {{
      const safeCandidates = Array.isArray(candidates) ? candidates : [];
      refs.candidateCount.textContent = `${{safeCandidates.length}} 项`;
      if (safeCandidates.length === 0) {{
        refs.candidateList.innerHTML = '<div class="candidate-item"><div class="muted">暂无候选结果，系统正在等待新画面。</div></div>';
        return;
      }}
      refs.candidateList.innerHTML = safeCandidates.map((candidate, index) => {{
        const label = escapeHtml(candidate.label || `candidate_${{index + 1}}`);
        const score = formatScore(candidate.score || 0);
        const summary = escapeHtml(candidate.summary || "暂无摘要");
        return `
          <div class="candidate-item">
            <div class="candidate-top">
              <div class="candidate-name">${{index + 1}}. ${{label}}</div>
              <div class="candidate-score">score ${{score}}</div>
            </div>
            <div class="candidate-summary">${{summary}}</div>
          </div>
        `;
      }}).join("");
    }}

    function renderMetrics(state) {{
      const metrics = {{
        frame_interval_seconds: state.frame_interval_seconds ?? 0,
        capture_seconds: state.capture_seconds ?? 0,
        analysis_seconds: state.analysis_seconds ?? 0,
        loop_seconds: state.loop_seconds ?? 0,
        ...state.latest_metrics,
      }};
      const entries = Object.entries(metrics);
      if (entries.length === 0) {{
        refs.metricGrid.innerHTML = '<div class="metric-chip"><div class="metric-key">status</div><div class="metric-value">暂无指标</div></div>';
        return;
      }}
      refs.metricGrid.innerHTML = entries.map(([key, value]) => {{
        const rendered = typeof value === "number" ? String(value) : JSON.stringify(value);
        return `
          <div class="metric-chip">
            <div class="metric-key">${{escapeHtml(key)}}</div>
            <div class="metric-value">${{escapeHtml(rendered)}}</div>
          </div>
        `;
      }}).join("");
    }}

    function renderNotes(notes) {{
      const safeNotes = Array.isArray(notes) ? notes : [];
      if (safeNotes.length === 0) {{
        refs.notesList.innerHTML = '<div class="note-pill">暂无补充说明</div>';
        return;
      }}
      refs.notesList.innerHTML = safeNotes
        .map((note) => `<div class="note-pill">${{escapeHtml(note)}}</div>`)
        .join("");
    }}

    function renderImage(state) {{
      const sourcePath = assetUrl(state.latest_capture_path || state.latest_annotated_path || "");
      if (!sourcePath) {{
        refs.mainImage.style.display = "none";
        refs.emptyImage.style.display = "grid";
        return;
      }}
      const token = `${{state.frame_index ?? 0}}-${{state.updated_at || ""}}`;
      if (refs.mainImage.dataset.token !== token) {{
        refs.mainImage.src = `${{sourcePath}}?v=${{encodeURIComponent(token)}}`;
        refs.mainImage.dataset.token = token;
      }}
      refs.mainImage.style.display = "block";
      refs.emptyImage.style.display = "none";
    }}

    function renderState(state) {{
      document.title = state.title || PAGE_TITLE;
      refs.pageTitle.textContent = state.title || PAGE_TITLE;
      refs.cameraBadge.textContent = "相机后端 " + (state.camera_backend || "待连接");
      refs.recognizerBadge.textContent = "识别后端 " + (state.recognizer_backend || "待连接");
      refs.updatedAt.textContent = "更新时间 " + (state.updated_at || "--");
      refs.stage.textContent = state.stage || "starting";
      refs.headline.textContent = state.headline || "等待视觉流启动";
      refs.detail.textContent = state.detail || "系统将自动突出当前最值得评委关注的目标。";
      refs.frameIndex.textContent = String(state.frame_index ?? 0);
      refs.captureSeconds.textContent = formatSeconds(state.capture_seconds);
      refs.analysisSeconds.textContent = formatSeconds(state.analysis_seconds);
      refs.loopSeconds.textContent = formatSeconds(state.loop_seconds);
      renderImage(state);
      renderTopCandidate(state.latest_candidates || []);
      renderCandidates(state.latest_candidates || []);
      renderMetrics(state);
      renderNotes(state.latest_notes || []);
    }}

    async function poll() {{
      try {{
        const response = await fetch(`${{JSON_URL}}?ts=${{Date.now()}}`, {{ cache: "no-store" }});
        if (!response.ok) {{
          return;
        }}
        const state = await response.json();
        renderState(state);
      }} catch (error) {{
        console.warn("vision status poll failed", error);
      }}
    }}

    renderState(bootstrap);
    poll();
    setInterval(poll, POLL_INTERVAL_MS);
  </script>
</body>
</html>
"""

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")
