"""Cached multilingual gTTS alerts for the simulation dashboard."""

from pathlib import Path
import hashlib

from . import config


def cached_voice(text: str, language: str = "en") -> Path:
    """Create or reuse a cached MP3; network failures are surfaced to the UI."""
    if language not in config.VOICE_LANGUAGES:
        raise ValueError(f"Unsupported voice language: {language}")
    config.VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(f"{language}:{text}".encode("utf-8")).hexdigest() + ".mp3"
    destination = config.VOICE_CACHE_DIR / filename
    if destination.exists():
        return destination
    try:
        from gtts import gTTS
        from gtts.tts import gTTSError
        from requests.exceptions import RequestException
        gTTS(text=text, lang=language).save(str(destination))
    except (OSError, ValueError, gTTSError, RequestException) as exc:
        raise RuntimeError("gTTS could not create the requested cached alert.") from exc
    return destination
