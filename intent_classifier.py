"""
intent_classifier.py — Jarvis intent detection
Classifies each user message into a structured Intent so the UI and
AgentExecutor know whether to use tools or just chat.

Intent types:
    chat            — plain conversation, no tools needed
    file_read       — user wants to read / open a specific file
    file_search     — user wants to search their PC for files/content
    file_index      — user wants to index a folder
    file_list       — user wants to list directory contents
    system_info     — CPU / RAM / disk / network stats
    open_app        — launch an application
    run_command     — execute a shell command
    web_search      — open a browser search through the safe tool layer
    memory_recall   — "do you remember…" style queries
    memory_clear    — clear memory
    voice_toggle    — enable / disable voice
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Intent:
    type: str                        # one of the strings above
    confidence: float = 1.0         # 0–1 rough confidence
    args: dict = field(default_factory=dict)  # extracted arguments

    def __repr__(self):
        return f"Intent({self.type}, {self.confidence:.2f}, {self.args})"


# ── Keyword pattern tables ─────────────────────────────────────────────────────

_FILE_READ_PATTERNS = [
    r"\bread\b.*(file|\.txt|\.py|\.md|\.json|\.csv|\.log)",
    r"\bopen\b.*(file|document|\.txt|\.py|\.md)",
    r"\bshow me\b.*(file|contents? of)",
    r"\bcat\b\s+\S+",
    r"\bprint\b.*(file|contents?)",
    r"\bload\b.*(file|document)",
]
_FILE_SEARCH_PATTERNS = [
    r"\b(find|search|look for|locate|where is)\b.*(file|folder|document|note|script)",
    r"\bfind files?\b",
    r"\bsearch my (pc|computer|drive|disk|files?)\b",
    r"\bwhich file\b",
    r"\bdo i have a file\b",
]
_FILE_LIST_PATTERNS = [
    r"\b(list|show|display)\b.*(files?|folder|directory|desktop|documents?)",
    r"\bwhat.*(files?|folder).*(in|on|at)\b",
    r"\bls\b",
    r"\bdir\b\s",
]
_FILE_INDEX_PATTERNS = [
    r"\b(index|scan|crawl|embed)\b.*(folder|directory|drive|pc|computer|files?)",
    r"\bbuild.*(file brain|index|knowledge base)\b",
    r"\bindex my (pc|computer|drive|documents?)\b",
]
_SYSTEM_INFO_PATTERNS = [
    r"\b(cpu|processor|ram|memory|disk|storage|battery|temperature|temp|gpu)\b.*(usage|load|percent|status|info|stats?)",
    r"\bhow much (ram|memory|cpu|disk)\b",
    r"\bwhat.?s (my )?(cpu|ram|disk|memory|battery)\b",
    r"\bsystem (info|stats?|status|diagnostics?)\b",
    r"\bresource (usage|monitor|stats?)\b",
]
_OPEN_APP_PATTERNS = [
    r"\b(open|launch|start|run|execute)\b.*(chrome|firefox|notepad|calculator|explorer|word|excel|spotify|vscode|terminal|cmd|powershell|paint)",
    r"\bopen (the )?(app|application|program|browser)\b",
    r"\bstart (the )?(app|application|program)\b",
]
_RUN_CMD_PATTERNS = [
    r"\b(run|execute|shell|cmd|bash|powershell)\b.*(command|script|\.bat|\.sh|\.ps1)",
    r"\brun this:?\s",
    r"\bexecute:?\s",
]
_WEB_SEARCH_PATTERNS = [
    r"\b(search|google|look up|find)\b.*\b(web|internet|online|google)\b",
    r"\bsearch (youtube|google|the web|the internet)\b",
    r"\bopen\b.*\b(website|url|web page)\b",
    r"\bhttps?://",
]
_MEMORY_RECALL_PATTERNS = [
    r"\bdo you remember\b",
    r"\bwhat did (i|we) (say|discuss|talk about)\b",
    r"\brecall\b.*(conversation|memory|earlier|before)",
    r"\bprevious (chat|conversation|session)\b",
    r"\blast time\b",
]
_MEMORY_CLEAR_PATTERNS = [
    r"\b(clear|delete|wipe|forget|erase)\b.*(memory|memories|history|conversations?)\b",
    r"\bforget everything\b",
    r"\breset memory\b",
]
_VOICE_TOGGLE_PATTERNS = [
    r"\b(enable|disable|turn on|turn off|start|stop)\b.*(voice|speech|tts|stt|speaking|listening)\b",
    r"\b(mute|unmute)\b.*(jarvis|voice|speech)\b",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(re.search(pat, low) for pat in patterns)


def _extract_path(text: str) -> str | None:
    """Pull the first file/folder path from the message."""
    m = re.search(r"['\"]?([~/\\][\w/\\. -]{3,}|[A-Za-z]:\\[\w/\\. -]{3,})['\"]?", text)
    return m.group(1) if m else None


def _extract_app(text: str) -> str | None:
    apps = ["chrome", "firefox", "notepad", "calculator", "explorer",
            "word", "excel", "spotify", "vscode", "terminal", "cmd",
            "powershell", "paint", "brave", "edge"]
    low = text.lower()
    for a in apps:
        if a in low:
            return a
    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(text: str) -> Intent:
    """
    Return an Intent for the given user message.
    Checks patterns in priority order; falls back to 'chat'.
    """
    if _match_any(text, _MEMORY_CLEAR_PATTERNS):
        return Intent("memory_clear", 0.95)

    if _match_any(text, _MEMORY_RECALL_PATTERNS):
        return Intent("memory_recall", 0.90)

    if _match_any(text, _FILE_INDEX_PATTERNS):
        path = _extract_path(text)
        return Intent("file_index", 0.90, {"path": path})

    if _match_any(text, _FILE_READ_PATTERNS):
        path = _extract_path(text)
        return Intent("file_read", 0.90, {"path": path})

    if _match_any(text, _FILE_LIST_PATTERNS):
        path = _extract_path(text)
        return Intent("file_list", 0.88, {"path": path})

    if _match_any(text, _FILE_SEARCH_PATTERNS):
        # Try to pull the search query — text after "find"/"search for" etc.
        m = re.search(r"(?:find|search for?|locate|look for)\s+(.+)", text, re.I)
        query = m.group(1).strip() if m else text
        return Intent("file_search", 0.88, {"query": query})

    if _match_any(text, _SYSTEM_INFO_PATTERNS):
        return Intent("system_info", 0.92)

    if _match_any(text, _OPEN_APP_PATTERNS):
        app = _extract_app(text)
        return Intent("open_app", 0.88, {"app": app})

    if _match_any(text, _RUN_CMD_PATTERNS):
        m = re.search(r"(?:run|execute|shell|cmd|bash)\s*:?\s*(.+)", text, re.I)
        cmd = m.group(1).strip() if m else ""
        return Intent("run_command", 0.80, {"command": cmd})

    if _match_any(text, _WEB_SEARCH_PATTERNS):
        m = re.search(
            r"(?:search|google|look up|find)\s+(?:(?:google|the web|the internet|online)\s+)?(?:for\s+)?(.+)",
            text,
            re.I,
        )
        query = m.group(1).strip() if m else text
        return Intent("web_search", 0.82, {"query": query})

    if _match_any(text, _VOICE_TOGGLE_PATTERNS):
        enable = any(w in text.lower() for w in ("enable", "turn on", "start", "unmute"))
        return Intent("voice_toggle", 0.85, {"enable": enable})

    return Intent("chat", 0.99)


def needs_agent(intent: Intent) -> bool:
    """Return True if this intent requires tool/agent execution."""
    return intent.type not in ("chat", "memory_recall")
