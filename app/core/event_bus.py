from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class AppEvent:
    kind: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    @property
    def stamp(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))


Subscriber = Callable[[AppEvent], None]


class EventBus:
    def __init__(self):
        self._lock = threading.RLock()
        self._subscribers: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(self, kind: str, summary: str, payload: dict[str, Any] | None = None) -> AppEvent:
        event = AppEvent(kind=kind, summary=summary, payload=dict(payload or {}))
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                continue
        return event
