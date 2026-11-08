"""Simulation-only action routing with alert cooldowns."""

from dataclasses import dataclass
from datetime import datetime, timezone

from . import config
from .voice import cached_voice


@dataclass(frozen=True)
class ActionEvent:
    level: str
    message: str
    payload: dict[str, object] | None = None
    audio_paths: tuple[str, ...] = ()


class ActionRouter:
    def __init__(self) -> None:
        self.last_alert_at: dict[str, float] = {}

    def route(self, level: str, zone: str, estimated_people: int, risk: float, timestamp: float) -> ActionEvent | None:
        if level == "Green":
            return None
        if timestamp - self.last_alert_at.get(level, -float("inf")) < config.ALERT_COOLDOWN_SECONDS:
            return None
        self.last_alert_at[level] = timestamp
        now = datetime.now(timezone.utc).isoformat()
        prefix = "CRITICAL " if level == "Red" else ""
        message = f"Simulation: {prefix}{level} alert in {zone} — risk {risk:.1f}."
        payload = None
        languages = ()
        if level == "Orange":
            languages = ("en", "hi", "te")
        elif level == "Red":
            languages = ("en", "hi", "te")
            payload = {
                "zone": zone,
                "estimated people": estimated_people,
                "risk": round(risk, 1),
                "timestamp": now,
            }
        audio_paths = []
        voice_error = None
        for language in languages:
            try:
                audio_paths.append(str(cached_voice(message, language)))
            except RuntimeError as exc:
                voice_error = str(exc)
        if voice_error:
            message = f"{message} Voice unavailable: {voice_error}"
        return ActionEvent(level, message, payload, tuple(audio_paths))
