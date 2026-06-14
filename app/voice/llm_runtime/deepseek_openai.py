import json
import os
import urllib.request


DEFAULT_MODEL = os.getenv("SPACEMIT_LLM_MODEL", "qwen2.5:0.5b")
DEFAULT_API_URL = os.getenv("SPACEMIT_OLLAMA_API", "http://127.0.0.1:11434/api/chat")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SPACEMIT_SYSTEM_PROMPT",
    "You are a board-side voice assistant. Reply directly in Chinese. "
    "Do not reveal chain-of-thought. Do not use bullet points. "
    "Keep the reply within two sentences, ideally under 30 Chinese characters.",
)
DEFAULT_MAX_TOKENS = int(os.getenv("SPACEMIT_MAX_TOKENS", "32"))
DEFAULT_TEMPERATURE = float(os.getenv("SPACEMIT_TEMPERATURE", "0"))
DEFAULT_CONTEXT = int(os.getenv("SPACEMIT_NUM_CTX", "1024"))
DEFAULT_KEEP_ALIVE = os.getenv("SPACEMIT_KEEP_ALIVE", "30m")
DEFAULT_TIMEOUT = int(os.getenv("SPACEMIT_LLM_TIMEOUT", "300"))
DEFAULT_MAX_CHARS = int(os.getenv("SPACEMIT_MAX_CHARS", "28"))
DEFAULT_MIN_CHARS = int(os.getenv("SPACEMIT_MIN_CHARS", "8"))
STOP_AFTER_FIRST_SENTENCE = os.getenv("SPACEMIT_STOP_AFTER_FIRST_SENTENCE", "1") == "1"


class ThinkTagFilter:
    def __init__(self):
        self.in_think = False
        self.tag_buffer = ""

    def feed(self, chunk):
        if not chunk:
            return ""

        text = self.tag_buffer + chunk
        self.tag_buffer = ""
        out = []
        i = 0
        while i < len(text):
            remaining = text[i:]
            if not self.in_think and remaining.startswith("<think>"):
                self.in_think = True
                i += len("<think>")
                continue
            if self.in_think and remaining.startswith("</think>"):
                self.in_think = False
                i += len("</think>")
                continue

            pending = False
            for tag in ("<think>", "</think>"):
                if tag.startswith(remaining):
                    self.tag_buffer = remaining
                    pending = True
                    break
            if pending:
                break

            if not self.in_think:
                out.append(text[i])
            i += 1
        return "".join(out)

    def flush(self):
        if self.in_think:
            self.tag_buffer = ""
            return ""
        tail = self.tag_buffer
        self.tag_buffer = ""
        return tail


class LlmModel:
    def __init__(self, model_path=None):
        self._model = model_path or DEFAULT_MODEL
        self._api_url = DEFAULT_API_URL
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        self._max_tokens = DEFAULT_MAX_TOKENS
        self._temperature = DEFAULT_TEMPERATURE
        self._num_ctx = DEFAULT_CONTEXT
        self._keep_alive = DEFAULT_KEEP_ALIVE
        self._timeout = DEFAULT_TIMEOUT
        self._max_chars = DEFAULT_MAX_CHARS
        self._min_chars = DEFAULT_MIN_CHARS
        self._stop_after_first_sentence = STOP_AFTER_FIRST_SENTENCE

    def _clip_visible_chunk(self, current_text, new_text):
        combined = current_text + new_text
        stop_at = None
        hard_endings = ".!?\u3002\uff01\uff1f;\uff1b"

        if self._stop_after_first_sentence and len(combined) >= self._min_chars:
            for idx, ch in enumerate(combined):
                if ch in hard_endings:
                    stop_at = idx + 1
                    break

        if stop_at is None and len(combined) >= self._max_chars:
            stop_at = self._max_chars

        if stop_at is None:
            return new_text, False

        clipped = combined[:stop_at]
        return clipped[len(current_text):], True

    def _payload(self, text):
        return {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "stream": True,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": self._max_tokens,
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
            },
        }

    def generate(self, text):
        payload = json.dumps(self._payload(text)).encode("utf-8")
        req = urllib.request.Request(
            self._api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        think_filter = ThinkTagFilter()
        visible_text = ""
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunk = obj.get("message", {}).get("content", "")
                visible = think_filter.feed(chunk)
                if visible:
                    clipped, should_stop = self._clip_visible_chunk(visible_text, visible)
                    if clipped:
                        visible_text += clipped
                        yield clipped
                    if should_stop:
                        break
                if obj.get("done"):
                    tail = think_filter.flush()
                    if tail:
                        clipped, _ = self._clip_visible_chunk(visible_text, tail)
                        if clipped:
                            yield clipped
                    break


if __name__ == "__main__":
    llm_model = LlmModel()
    for c in llm_model.generate("Please introduce yourself in one short Chinese sentence."):
        print(c, end="", flush=True)
