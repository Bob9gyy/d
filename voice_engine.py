"""
voice_engine.py — Jarvis voice pipeline
Provides:
  • Wake-word detection ("hey jarvis" / "jarvis") via Vosk (offline STT)
  • Continuous speech-to-text transcription (Vosk)
  • Text-to-speech output (pyttsx3 — fully offline)
  • Simple callback API so main_ui.py can integrate without blocking Qt

Dependencies (all offline):
    pip install vosk sounddevice pyttsx3

Vosk model (download once, ~40 MB):
    https://alphacephei.com/vosk/models  →  vosk-model-small-en-us-0.15
    Unzip to ./vosk-model-small-en-us-0.15/  (next to this file)

USAGE:
    from voice_engine import VoiceEngine

    def on_wake():
        print("Wake word detected!")

    def on_transcript(text):
        print(f"Heard: {text}")

    engine = VoiceEngine(on_wake=on_wake, on_transcript=on_transcript)
    engine.start()          # non-blocking background thread
    engine.speak("Hello")   # TTS
    engine.stop()
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

# ── Optional heavy imports ─────────────────────────────────────────────────────

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

try:
    from vosk import Model, KaldiRecognizer
    _VOSK_OK = True
except ImportError:
    _VOSK_OK = False

try:
    import pyttsx3
    _TTS_OK = True
except ImportError:
    _TTS_OK = False

# ── Configuration ──────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16000
BLOCK_SIZE    = 8000    # frames per audio block
WAKE_WORDS    = {"hey jarvis", "jarvis", "hey jarvis wake up"}
MODEL_DIRS    = [
    "./vosk-model-small-en-us-0.15",
    "./vosk-model-en-us-0.22",
    "~/vosk-model-small-en-us-0.15",
    "~/vosk-model-en-us-0.22",
]

# TTS voice settings
TTS_RATE    = 175   # words per minute
TTS_VOLUME  = 0.92


class VoiceEngine:
    """
    Offline voice pipeline: wake word → STT → callback, plus TTS output.

    Args:
        on_wake:       Called (on audio thread) when a wake word is detected.
        on_transcript: Called with the final transcription string.
        on_status:     Called with human-readable status strings (for UI).
        model_path:    Path to Vosk model directory (auto-detected if None).
    """

    def __init__(
        self,
        on_wake:       Optional[Callable[[], None]] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_status:     Optional[Callable[[str], None]] = None,
        model_path:    Optional[str] = None,
    ):
        self.on_wake       = on_wake       or (lambda: None)
        self.on_transcript = on_transcript or (lambda t: None)
        self.on_status     = on_status     or (lambda s: print(f"[Voice] {s}"))

        self.available   = _SD_OK and _VOSK_OK and _TTS_OK
        self._running    = False
        self._listening  = False   # active capture after wake word
        self._thread: Optional[threading.Thread] = None
        self._tts_lock   = threading.Lock()
        self._audio_q: queue.Queue = queue.Queue()

        self._model:     Optional[Model]          = None
        self._rec:       Optional[KaldiRecognizer] = None
        self._tts_engine = None

        if not self.available:
            missing = []
            if not _SD_OK:   missing.append("sounddevice")
            if not _VOSK_OK: missing.append("vosk")
            if not _TTS_OK:  missing.append("pyttsx3")
            self.on_status(
                f"Voice disabled — missing packages: {', '.join(missing)}\n"
                f"Install:  pip install {' '.join(missing)}"
            )
            return

        # Locate Vosk model
        mp = model_path or self._find_model()
        if mp is None:
            self.available = False
            self.on_status(
                "Voice disabled — Vosk model not found.\n"
                "Download: https://alphacephei.com/vosk/models\n"
                "Unzip vosk-model-small-en-us-0.15 next to voice_engine.py"
            )
            return

        try:
            self._model = Model(mp)
            self._rec   = KaldiRecognizer(self._model, SAMPLE_RATE)
            self.on_status(f"Vosk model loaded: {mp}")
        except Exception as exc:
            self.available = False
            self.on_status(f"Vosk model error: {exc}")
            return

        try:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate",   TTS_RATE)
            self._tts_engine.setProperty("volume", TTS_VOLUME)
            # Try to pick a natural-sounding voice
            voices = self._tts_engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower() or "david" in v.name.lower():
                    self._tts_engine.setProperty("voice", v.id)
                    break
            self.on_status("TTS engine ready.")
        except Exception as exc:
            self.on_status(f"TTS init warning: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the background voice thread.  Returns False if unavailable."""
        if not self.available:
            return False
        if self._running:
            return True
        self._running = True
        self._thread  = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        self.on_status("Listening for wake word …")
        return True

    def stop(self):
        """Stop the voice pipeline."""
        self._running   = False
        self._listening = False
        if self._thread:
            self._thread.join(timeout=3)
        self.on_status("Voice stopped.")

    def speak(self, text: str):
        """
        Speak text via TTS (blocking, runs on a daemon thread to avoid
        blocking the Qt main thread).
        """
        if not self.available or self._tts_engine is None:
            return

        def _do_speak():
            with self._tts_lock:
                try:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                except Exception as exc:
                    self.on_status(f"TTS error: {exc}")

        threading.Thread(target=_do_speak, daemon=True).start()

    def set_listening(self, active: bool):
        """
        Manually start/stop active transcription mode
        (e.g. called from UI when push-to-talk button held).
        """
        self._listening = active

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal audio loop ────────────────────────────────────────────────────

    def _find_model(self) -> Optional[str]:
        for mp in MODEL_DIRS:
            p = Path(mp).expanduser()
            if p.exists() and p.is_dir():
                return str(p)
        return None

    def _audio_callback(self, indata, frames, time, status):
        """sounddevice callback — puts raw bytes into queue."""
        if status:
            pass  # ignore overflow/underflow silently
        self._audio_q.put(bytes(indata))

    def _listen_loop(self):
        """
        Background thread: continuously records audio and feeds Vosk.
        Wake-word mode → when triggered, enters active transcription.
        """
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                self.on_status("Microphone open.")
                while self._running:
                    try:
                        data = self._audio_q.get(timeout=1)
                    except queue.Empty:
                        continue

                    if self._rec.AcceptWaveform(data):
                        result  = json.loads(self._rec.Result())
                        phrase  = result.get("text", "").strip().lower()

                        if not phrase:
                            continue

                        if not self._listening:
                            # Wake-word detection mode
                            for ww in WAKE_WORDS:
                                if ww in phrase:
                                    self._listening = True
                                    self.on_wake()
                                    break
                        else:
                            # Active transcription — emit the phrase
                            if phrase:
                                self.on_transcript(phrase)
                            # Auto-exit active mode after one utterance
                            self._listening = False
                            self.on_status("Listening for wake word …")

        except Exception as exc:
            self.on_status(f"Audio error: {exc}")
            self._running = False


# ── Qt-friendly wrapper ────────────────────────────────────────────────────────

class VoiceSignalBridge:
    """
    Thin bridge that converts VoiceEngine callbacks into Qt Signals.
    Instantiate this in JarvisWindow and pass its methods as callbacks.

    Example:
        bridge  = VoiceSignalBridge(self)
        engine  = VoiceEngine(
            on_wake=bridge.on_wake,
            on_transcript=bridge.on_transcript,
            on_status=bridge.on_status,
        )
    """
    from PySide6.QtCore import QObject, Signal

    class _Signals(QObject):
        wake       = Signal()
        transcript = Signal(str)
        status     = Signal(str)

    def __init__(self, parent=None):
        self._sig = self._Signals(parent)

    # Connect your slots to these:
    @property
    def wake_signal(self):       return self._sig.wake
    @property
    def transcript_signal(self): return self._sig.transcript
    @property
    def status_signal(self):     return self._sig.status

    # Pass these as VoiceEngine callbacks:
    def on_wake(self):               self._sig.wake.emit()
    def on_transcript(self, t: str): self._sig.transcript.emit(t)
    def on_status(self, s: str):     self._sig.status.emit(s)


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    def wake():
        print("\n🎤  Wake word detected!  Listening …")

    def heard(text):
        print(f"\n📝  Transcribed: {text}")

    engine = VoiceEngine(on_wake=wake, on_transcript=heard)
    if engine.start():
        print("Voice engine running.  Say 'Hey Jarvis' …  Ctrl+C to quit.")
        engine.speak("JARVIS online.  Awaiting command.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            engine.stop()
    else:
        print("Voice engine unavailable — check dependencies.")
