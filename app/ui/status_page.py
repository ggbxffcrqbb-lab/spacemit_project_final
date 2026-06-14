from __future__ import annotations

from collections import deque
from html import escape
import json
import threading
import time


class StatusPageWriter:
    def __init__(self, config, project_name: str, llm_model: str, rag_enabled: bool):
        self.config = config
        self.project_name = project_name
        self.llm_model = llm_model
        self.rag_enabled = rag_enabled
        self._lock = threading.RLock()
        self._history = deque(maxlen=config.history_limit)
        self._state = {
            "project_name": project_name,
            "llm_model": llm_model,
            "rag_enabled": rag_enabled,
            "stage": "starting",
            "headline": "正在启动",
            "detail": "准备板端语音与知识库服务",
            "latest_user_text": "",
            "latest_reply_text": "",
            "latest_citations": [],
            "latest_metrics": {},
            "latest_rag_hits": [],
            "updated_at": self._now(),
        }
        self.config.status_dir.mkdir(parents=True, exist_ok=True)
        self._render_locked()

    def update(
        self,
        stage: str,
        headline: str,
        detail: str = "",
        latest_user_text: str | None = None,
        latest_reply_text: str | None = None,
        latest_citations: list[str] | None = None,
        latest_metrics: dict | None = None,
        latest_rag_hits: list[dict] | None = None,
    ):
        with self._lock:
            self._state["stage"] = stage
            self._state["headline"] = headline
            self._state["detail"] = detail
            self._state["updated_at"] = self._now()
            if latest_user_text is not None:
                self._state["latest_user_text"] = latest_user_text
            if latest_reply_text is not None:
                self._state["latest_reply_text"] = latest_reply_text
            if latest_citations is not None:
                self._state["latest_citations"] = latest_citations
            if latest_metrics is not None:
                self._state["latest_metrics"] = latest_metrics
            if latest_rag_hits is not None:
                self._state["latest_rag_hits"] = latest_rag_hits
            self._render_locked()

    def record_turn(self, result):
        with self._lock:
            self._history.appendleft(
                {
                    "ts": self._now(),
                    "user_text": result.user_text,
                    "reply_text": result.reply_text,
                    "citations": list(result.citations),
                    "metrics": {
                        "first_chunk_ms": result.first_chunk_ms,
                        "first_tts_enqueue_ms": result.first_tts_enqueue_ms,
                        "total_ms": result.total_ms,
                        "output_chars": result.output_chars,
                        "rag_used": result.rag_used,
                    },
                }
            )
            self._render_locked()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "status_dir": str(self.config.status_dir),
                "html_path": str(self.config.html_path),
                "json_path": str(self.config.json_path),
                "text_path": str(self.config.text_path),
                "stage": self._state["stage"],
                "updated_at": self._state["updated_at"],
                "history_count": len(self._history),
            }

    def _render_locked(self):
        snapshot = {
            "state": dict(self._state),
            "history": list(self._history),
        }
        self.config.json_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.config.text_path.write_text(self._render_text(snapshot), encoding="utf-8")
        self.config.html_path.write_text(self._render_html(snapshot), encoding="utf-8")

    def _render_text(self, snapshot: dict) -> str:
        state = snapshot["state"]
        lines = [
            self.config.title,
            "=" * len(self.config.title),
            f"阶段：{state['stage']}",
            f"状态：{state['headline']}",
            f"说明：{state['detail']}",
            f"模型：{state['llm_model']}",
            f"RAG：{'开启' if state['rag_enabled'] else '关闭'}",
            f"更新时间：{state['updated_at']}",
            "",
            "最近问题：",
            state["latest_user_text"] or "暂无",
            "",
            "最近回答：",
            state["latest_reply_text"] or "暂无",
        ]
        if state["latest_citations"]:
            lines.extend(["", "最近引用：", *state["latest_citations"]])
        if state["latest_rag_hits"]:
            lines.append("")
            lines.append("最近检索命中：")
            for hit in state["latest_rag_hits"]:
                label = hit.get("source_label", "")
                lines.append(f"- {hit.get('title', '未命名')} | {label} | score={hit.get('score')}")
        if state["latest_metrics"]:
            lines.append("")
            lines.append("最近指标：")
            for key, value in state["latest_metrics"].items():
                lines.append(f"- {key}: {value}")

        history = snapshot["history"][: self.config.history_limit]
        if history:
            lines.append("")
            lines.append("最近记录：")
            for index, item in enumerate(history, start=1):
                lines.append(f"{index}. {item['ts']} | {item['user_text']}")
        return "\n".join(lines) + "\n"

    def _render_html(self, snapshot: dict) -> str:
        state = snapshot["state"]
        history_cards = "".join(
            f"""
            <section class="history-card">
              <div class="history-meta">{escape(item['ts'])}</div>
              <div class="history-q">Q: {escape(item['user_text'])}</div>
              <div class="history-a">A: {escape(item['reply_text'][:220])}</div>
            </section>
            """
            for item in snapshot["history"][: self.config.history_limit]
        )
        citations = "".join(
            f"<li>{escape(citation)}</li>" for citation in state.get("latest_citations", [])
        ) or "<li>暂无</li>"
        rag_hits = "".join(
            f"<li>{escape(hit['title'])} | {escape(hit.get('source_label', ''))} | score={hit['score']}</li>"
            for hit in state.get("latest_rag_hits", [])
        ) or "<li>暂无</li>"
        metrics = state.get("latest_metrics", {})
        metrics_html = "".join(
            f"<li>{escape(str(key))}: {escape(str(value))}</li>"
            for key, value in metrics.items()
        ) or "<li>暂无</li>"

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{int(self.config.refresh_seconds)}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(self.config.title)}</title>
  <style>
    :root {{
      --bg: #0f1720;
      --panel: #172331;
      --line: #284255;
      --ink: #edf4fb;
      --muted: #9bb2c8;
      --accent: #4db6ac;
      --warn: #f2b950;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(77,182,172,0.18), transparent 28%),
        linear-gradient(180deg, #0b141d, var(--bg));
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .panel {{
      background: rgba(23,35,49,0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 18px 50px rgba(0,0,0,0.25);
    }}
    .title {{
      font-size: 34px;
      font-weight: 700;
      letter-spacing: 1px;
      margin: 0 0 10px;
    }}
    .stage {{
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(77,182,172,0.16);
      color: #d6fffb;
      border: 1px solid rgba(77,182,172,0.4);
      font-weight: 600;
      margin-bottom: 12px;
    }}
    .headline {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    .detail {{
      color: var(--muted);
      line-height: 1.7;
      margin: 0;
      white-space: pre-wrap;
    }}
    .meta-list, .mini-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.8;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .section-title {{
      margin: 0 0 12px;
      font-size: 18px;
      color: #e7f4ff;
    }}
    .content {{
      white-space: pre-wrap;
      line-height: 1.8;
      font-size: 18px;
    }}
    .history {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .history-card {{
      border: 1px solid rgba(77,182,172,0.18);
      border-radius: 16px;
      padding: 14px;
      background: rgba(8,17,24,0.24);
    }}
    .history-meta {{
      color: var(--warn);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .history-q {{
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .history-a {{
      color: var(--muted);
      line-height: 1.7;
    }}
    @media (max-width: 900px) {{
      .hero, .grid, .history {{
        grid-template-columns: 1fr;
      }}
      .wrap {{
        padding: 16px;
      }}
      .title {{
        font-size: 28px;
      }}
      .headline {{
        font-size: 24px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <section class="panel">
        <h1 class="title">{escape(self.config.title)}</h1>
        <div class="stage">{escape(state['stage'])}</div>
        <h2 class="headline">{escape(state['headline'])}</h2>
        <p class="detail">{escape(state['detail'])}</p>
      </section>
      <section class="panel">
        <h3 class="section-title">运行概况</h3>
        <ul class="meta-list">
          <li>项目：{escape(state['project_name'])}</li>
          <li>模型：{escape(state['llm_model'])}</li>
          <li>RAG：{"开启" if state['rag_enabled'] else "关闭"}</li>
          <li>更新时间：{escape(state['updated_at'])}</li>
          <li>HTML：{escape(str(self.config.html_path))}</li>
        </ul>
      </section>
    </div>
    <div class="grid">
      <section class="panel">
        <h3 class="section-title">最近问题</h3>
        <div class="content">{escape(state['latest_user_text'] or '暂无')}</div>
      </section>
      <section class="panel">
        <h3 class="section-title">最近回答</h3>
        <div class="content">{escape(state['latest_reply_text'] or '暂无')}</div>
      </section>
      <section class="panel">
        <h3 class="section-title">最近引用</h3>
        <ul class="mini-list">{citations}</ul>
      </section>
      <section class="panel">
        <h3 class="section-title">最近检索命中</h3>
        <ul class="mini-list">{rag_hits}</ul>
      </section>
      <section class="panel">
        <h3 class="section-title">最近指标</h3>
        <ul class="mini-list">{metrics_html}</ul>
      </section>
    </div>
    <section class="panel">
      <h3 class="section-title">最近问答记录</h3>
      <div class="history">{history_cards or '<div class="detail">暂无</div>'}</div>
    </section>
  </div>
</body>
</html>
"""

    def _now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")
