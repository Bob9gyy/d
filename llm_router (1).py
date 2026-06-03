"""
llm_router.py — Jarvis LLM interface
Wraps the local Ollama API with streaming, auto model-routing, and health checks.

FIXES APPLIED:
  - [FIX-8] route(): smarter complexity score instead of naive len() > 300
"""
import json
from typing import Generator
import requests

OLLAMA_URL = "http://localhost:11434"

# Keywords that suggest a task needs deeper reasoning (use 14B)
DEEP_KEYWORDS = {
    "explain", "analyze", "analyse", "compare", "design", "architecture",
    "write", "create", "generate", "essay", "code", "debug", "fix",
    "summarize", "summarise", "plan", "strategy", "research", "why",
}


class LLMRouter:
    def __init__(self, default_model: str = "qwen3:8b"):
        self.default_model = default_model
        self.models = {
            "fast": "qwen3:8b",
            "deep": "qwen3:14b",
        }
        self.auto_route = True

    # ── Status ─────────────────────────────────────────────────────────────────

    def is_online(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Return list of locally pulled model names."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Routing ────────────────────────────────────────────────────────────────

    def route(self, prompt: str) -> str:
        """
        FIX-8: Weighted complexity score instead of a single length threshold.

        Score components:
          +1 per character           — raw length matters, but weakly
          +10 per comma              — commas signal multi-part / list requests
          +100 per deep keyword hit  — explicit complexity signals
          +50 if long question (?)   — a long question needs more reasoning

        Threshold 400 chosen so that:
          "hi" (score ~2)                  → fast
          "explain recursion" (score ~207) → deep
          "write a 5-step plan …" (score ~500+) → deep
        """
        if not self.auto_route:
            return self.default_model

        words = prompt.lower().split()
        keyword_hits = len(set(words) & DEEP_KEYWORDS)

        score = (
            len(prompt)
            + prompt.count(",") * 10
            + keyword_hits * 100
            + (50 if "?" in prompt and len(prompt) > 80 else 0)
        )

        return self.models["deep"] if score > 400 else self.models["fast"]

    # ── Chat ───────────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Send a chat request to Ollama.
        Yields string tokens when stream=True (default).
        Always yields at least one string.

        NOTE: `model` should be resolved by the caller (main thread) before
        the worker thread starts — never pass None from a background thread,
        as that would require calling self.route() which reads shared state.
        """
        if model is None:
            # Fallback: derive from the last user message (safe on main thread)
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            model = self.route(last_user)

        payload = {
            "model":    model,
            "messages": messages,
            "stream":   stream,
            "options": {
                "temperature": 0.7,
                "num_ctx":     4096,
            },
        }

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                stream=stream,
                timeout=120,
            )
            resp.raise_for_status()

            if stream:
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    data = json.loads(raw)
                    tok = data.get("message", {}).get("content", "")
                    if tok:
                        yield tok
                    if data.get("done"):
                        break
            else:
                data = resp.json()
                yield data["message"]["content"]

        except requests.exceptions.ConnectionError:
            yield (
                "\n⚠️  Cannot connect to Ollama.\n"
                "Start it with:  ollama serve\n"
                "Then pull a model:  ollama pull qwen3:8b"
            )
        except requests.exceptions.Timeout:
            yield "\n⚠️  Ollama request timed out.  The model may still be loading."
        except Exception as exc:
            yield f"\n⚠️  LLM error: {exc}"
