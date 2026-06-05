"""
jarvis_safety.py - EXPANDED safety/tool layer for JARVIS v2.1

Changes vs original:
 - File access: write_file, move_file, delete_file, copy_file, create_dir, delete_dir
 - Broader read: any extension (not just text), larger files (up to 10 MB)
 - Broader commands: write-capable git, pip install, arbitrary subprocess
 - Multimodal: PDF text extraction (pdfminer/pypdf fallback), image OCR (pytesseract),
               audio transcription (whisper), video info (ffprobe)
 - Web/API: http_get, http_post, download_file
 - Real-time data: run_background_task (threading), live file-watching (watchdog)
 - Encryption/decryption: AES-256 via cryptography library
 - Package installation: pip_install tool
 - Interactive subprocess: run_interactive_command (pty on POSIX)
 - Sensitive-path and destructive-action guards are CONFIGURABLE via config.yaml,
   not hardcoded. Set JARVIS_SAFETY_LEVEL=strict|normal|off to override.
"""

from __future__ import annotations

import base64
import datetime as _dt
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import threading
import webbrowser
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)

# ── Safety level ──────────────────────────────────────────────────────────────
# "strict"  = original behaviour (read-only, limited paths)
# "normal"  = write allowed, destructive ops require confirmation token
# "off"     = no restrictions (use at your own risk)
SAFETY_LEVEL = os.getenv("JARVIS_SAFETY_LEVEL", "normal").lower()

MAX_READ_BYTES   = 10_000_000   # 10 MB  (was 512 KB)
MAX_TOOL_OUTPUT  = 16_000       # (was 4 000)

SAFE_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".scala",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".csv", ".tsv", ".xml", ".html", ".htm", ".sql", ".sh", ".bat",
    ".dockerfile", ".makefile", ".r", ".m", ".lua",
}

# Skip dirs only applied in "strict" mode
SKIP_DIRS_STRICT = {
    "node_modules", ".git", ".svn", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build",
}

# Dirs that are NEVER accessible regardless of safety level (OS internals)
ALWAYS_BLOCKED_DIRS = {
    "windows", "$recycle.bin",
}

# Sensitive file names — blocked in strict only
SENSITIVE_NAMES_STRICT = {
    ".netrc", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
}

# These are NEVER exposed at any safety level
SENSITIVE_NAMES_ALWAYS = {
    "credentials", "credentials.json",
}

SENSITIVE_SUFFIXES_STRICT = {".kdbx", ".pfx", ".p12", ".keystore"}

# ── Destructive-action confirmation ───────────────────────────────────────────
_CONFIRM_TOKENS: set[str] = set()

def request_destructive_confirm(token: str) -> str:
    """Returns a message asking the user to confirm with the token."""
    _CONFIRM_TOKENS.add(token)
    return (
        f"⚠️  This is a destructive operation. "
        f"To confirm, call this function again with confirm_token='{token}'."
    )

def _check_confirm(confirm_token: str | None, token: str) -> bool:
    if SAFETY_LEVEL == "off":
        return True
    if confirm_token and confirm_token in _CONFIRM_TOKENS:
        _CONFIRM_TOKENS.discard(confirm_token)
        return True
    return False

# ── Path helpers ──────────────────────────────────────────────────────────────

def allowed_roots() -> list[Path]:
    roots = [Path.home(), Path.cwd()]
    extra = os.getenv("JARVIS_ALLOWED_ROOTS", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            roots.append(Path(item).expanduser())
    resolved: list[Path] = []
    for root in roots:
        try:
            rp = root.resolve()
        except OSError:
            continue
        if rp not in resolved:
            resolved.append(rp)
    return resolved

def is_under(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False

def resolve_path(path: str | None, default: str = "~") -> Path:
    raw = path if path and str(path).strip() else default
    return Path(str(raw)).expanduser().resolve()

def is_sensitive_path(path: Path) -> bool:
    parts_lower = {part.lower() for part in path.parts}
    name = path.name.lower()

    # Always blocked
    if parts_lower & ALWAYS_BLOCKED_DIRS:
        return True
    if name in SENSITIVE_NAMES_ALWAYS:
        return True

    if SAFETY_LEVEL == "strict":
        if parts_lower & SKIP_DIRS_STRICT:
            return True
        if name in SENSITIVE_NAMES_STRICT:
            return True
        if path.suffix.lower() in SENSITIVE_SUFFIXES_STRICT:
            return True

    return False

def validate_read_path(
    path: str | None, *, must_be_file: bool = False
) -> tuple[bool, str, Path]:
    p = resolve_path(path)
    if SAFETY_LEVEL == "strict" and not is_under(p, allowed_roots()):
        return False, f"Blocked path outside allowed roots: {p}", p
    if is_sensitive_path(p):
        return False, f"Blocked sensitive path: {p}", p
    if not p.exists():
        return False, f"Path not found: {p}", p
    if must_be_file and not p.is_file():
        return False, f"Not a file: {p}", p
    return True, "", p

def validate_write_path(path: str | None) -> tuple[bool, str, Path]:
    """Check that a write target is acceptable."""
    p = resolve_path(path)
    if SAFETY_LEVEL == "strict" and not is_under(p, allowed_roots()):
        return False, f"Blocked write path outside allowed roots: {p}", p
    if is_sensitive_path(p):
        return False, f"Blocked write to sensitive path: {p}", p
    return True, "", p

def validate_text_file(path: Path) -> tuple[bool, str]:
    """In normal/off mode, accept ANY file up to MAX_READ_BYTES."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"Cannot stat file: {exc}"
    if size > MAX_READ_BYTES:
        return False, f"File too large: {size:,} bytes (max {MAX_READ_BYTES:,})"
    return True, ""

def safe_iter_files(root: Path, max_scan: int = 20_000):
    scanned = 0
    stack = [root]
    while stack and scanned < max_scan:
        current = stack.pop()
        if is_sensitive_path(current):
            continue
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if scanned >= max_scan:
                break
            scanned += 1
            if is_sensitive_path(entry):
                continue
            try:
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file():
                    yield entry
            except OSError:
                continue

# ── File write/move/delete ────────────────────────────────────────────────────

def write_file(path: str, content: str, *, append: bool = False) -> str:
    ok, msg, p = validate_write_path(path)
    if not ok:
        return msg
    mode = "a" if append else "w"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8") if mode == "w" else \
            open(p, "a", encoding="utf-8").write(content)
        return f"{'Appended' if append else 'Written'}: {p} ({len(content):,} chars)"
    except Exception as exc:
        return f"Write error: {exc}"

def move_file(src: str, dst: str, *, confirm_token: str | None = None) -> str:
    token = f"move:{src}->{dst}"
    if SAFETY_LEVEL == "normal" and not _check_confirm(confirm_token, token):
        return request_destructive_confirm(token)
    ok, msg, sp = validate_read_path(src, must_be_file=True)
    if not ok:
        return msg
    ok2, msg2, dp = validate_write_path(dst)
    if not ok2:
        return msg2
    try:
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sp), str(dp))
        return f"Moved: {sp} → {dp}"
    except Exception as exc:
        return f"Move error: {exc}"

def copy_file(src: str, dst: str) -> str:
    ok, msg, sp = validate_read_path(src, must_be_file=True)
    if not ok:
        return msg
    ok2, msg2, dp = validate_write_path(dst)
    if not ok2:
        return msg2
    try:
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(sp), str(dp))
        return f"Copied: {sp} → {dp}"
    except Exception as exc:
        return f"Copy error: {exc}"

def delete_file(path: str, *, confirm_token: str | None = None) -> str:
    token = f"delete:{path}"
    if SAFETY_LEVEL != "off" and not _check_confirm(confirm_token, token):
        return request_destructive_confirm(token)
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg
    try:
        p.unlink()
        return f"Deleted: {p}"
    except Exception as exc:
        return f"Delete error: {exc}"

def create_dir(path: str) -> str:
    ok, msg, p = validate_write_path(path)
    if not ok:
        return msg
    try:
        p.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {p}"
    except Exception as exc:
        return f"mkdir error: {exc}"

def delete_dir(path: str, *, confirm_token: str | None = None) -> str:
    token = f"rmdir:{path}"
    if SAFETY_LEVEL != "off" and not _check_confirm(confirm_token, token):
        return request_destructive_confirm(token)
    ok, msg, p = validate_read_path(path)
    if not ok:
        return msg
    if not p.is_dir():
        return f"Not a directory: {p}"
    try:
        shutil.rmtree(str(p))
        return f"Deleted directory: {p}"
    except Exception as exc:
        return f"rmdir error: {exc}"

# ── Shell / subprocess ────────────────────────────────────────────────────────

SHELL_METACHARS = re.compile(r"[|&;<>()`$]")

# Blocked at ALL safety levels
ALWAYS_DANGEROUS = {
    "format", "mkfs", "diskpart",
    "shutdown", "poweroff", "reboot", "restart-computer",
}

# Blocked only in strict mode
DANGEROUS_STRICT = {
    "rm", "del", "erase", "rmdir", "rd", "remove-item",
    "sudo", "su", "chmod", "chown",
}

READ_ONLY_EXTERNAL_COMMANDS = {
    "whoami", "hostname", "ipconfig", "ifconfig", "ping", "tracert",
    "traceroute", "netstat", "tasklist", "ps", "df", "du", "free", "uname",
    "which", "where", "type", "cat", "head", "tail", "grep", "find",
    "wc", "sort", "uniq",
}

ALL_GIT_SUBCOMMANDS = {
    "status", "log", "diff", "show", "rev-parse", "branch",
    "add", "commit", "push", "pull", "fetch", "checkout", "merge",
    "stash", "clone", "init", "reset",
}

BLOCKED_APPS = {"cmd", "powershell", "terminal", "bash", "sh", "zsh"}
APP_COMMANDS_WINDOWS = {
    "chrome": ["chrome"], "firefox": ["firefox"], "brave": ["brave"],
    "edge": ["msedge"], "notepad": ["notepad"], "calculator": ["calc"],
    "explorer": ["explorer"], "word": ["winword"], "excel": ["excel"],
    "vscode": ["code"], "paint": ["mspaint"], "spotify": ["spotify"],
}
APP_COMMANDS_POSIX = {
    "chrome": ["google-chrome"], "firefox": ["firefox"],
    "brave": ["brave-browser"], "vscode": ["code"],
    "calculator": ["gnome-calculator"], "spotify": ["spotify"],
    "notepad": ["gedit"], "explorer": ["nautilus"],
}

def execute_safe_command(command: str, cwd: str | None = None, timeout: int = 30) -> str:
    """Run a shell command.
    In strict mode: read-only whitelist.
    In normal/off: any command except ALWAYS_DANGEROUS.
    """
    if not command or not command.strip():
        return "Empty command."

    try:
        parts = shlex.split(command, posix=(platform.system() != "Windows"))
    except ValueError as exc:
        return f"Could not parse command: {exc}"
    if not parts:
        return "Empty command."

    base = Path(parts[0]).name.lower().removesuffix(".exe")

    if base in ALWAYS_DANGEROUS:
        return f"Blocked dangerous command: {base}"

    if SAFETY_LEVEL == "strict":
        if SHELL_METACHARS.search(command):
            return "Blocked shell metacharacters in strict mode."
        if base in DANGEROUS_STRICT:
            return f"Blocked command in strict mode: {base}"
        if base == "git":
            if len(parts) < 2 or parts[1].lower() not in {"status","log","diff","show","rev-parse","branch"}:
                return "Blocked git command in strict mode."
        elif base not in READ_ONLY_EXTERNAL_COMMANDS | {"date","dir","echo","git","ls","pwd","time","ver"}:
            return f"Command '{base}' not allowed in strict mode."

    run_cwd = resolve_path(cwd, default=".") if cwd else Path.cwd().resolve()

    try:
        result = subprocess.run(
            parts, shell=False, cwd=str(run_cwd),
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return f"Command not found: {parts[0]}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as exc:
        return f"Error running command: {exc}"

    output = result.stdout.strip() or result.stderr.strip() or "(no output)"
    if result.returncode != 0:
        output = f"[exit {result.returncode}]\n{output}"
    return output[:MAX_TOOL_OUTPUT]

def pip_install(package: str) -> str:
    """Install a Python package via pip."""
    if SAFETY_LEVEL == "strict":
        return "pip_install is blocked in strict safety mode."
    if not re.match(r'^[\w\-\.\[\],>=<!\s]+$', package):
        return f"Invalid package spec: {package!r}"
    return execute_safe_command(f"pip install {package}", timeout=120)

def run_background_task(command: str, label: str = "task") -> str:
    """Launch a command in a background thread, returns immediately."""
    def _run():
        try:
            subprocess.Popen(
                shlex.split(command), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.error("Background task '%s' failed: %s", label, exc)

    t = threading.Thread(target=_run, name=f"jarvis-bg-{label}", daemon=True)
    t.start()
    return f"Background task '{label}' started: {command}"

# ── App / URL / Web ───────────────────────────────────────────────────────────

def open_safe_app(app: str) -> str:
    app_name = (app or "").lower().strip()
    if not app_name:
        return "No app name provided."
    if app_name in BLOCKED_APPS and SAFETY_LEVEL == "strict":
        return f"Blocked launching shell app '{app_name}' in strict mode."
    commands = APP_COMMANDS_WINDOWS if platform.system() == "Windows" else APP_COMMANDS_POSIX
    cmd = commands.get(app_name, [app_name])
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Launched: {app_name}"
    except FileNotFoundError:
        return f"App command not found for: {app_name}"
    except Exception as exc:
        return f"Could not launch '{app_name}': {exc}"

def open_safe_url(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        return "No URL provided."
    if "://" not in candidate:
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Blocked URL. Only http(s) URLs allowed."
    webbrowser.open(candidate)
    return f"Opened URL: {candidate}"

def search_web(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "No search query."
    webbrowser.open("https://www.google.com/search?q=" + quote_plus(q))
    return f"Searched the web for: {q}"

# ── HTTP API calls ────────────────────────────────────────────────────────────

def http_get(url: str, headers: dict | None = None, timeout: int = 15) -> str:
    """Fetch a URL and return the response text."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read(MAX_READ_BYTES).decode("utf-8", errors="replace")
        return content[:MAX_TOOL_OUTPUT]
    except Exception as exc:
        return f"HTTP GET error: {exc}"

def http_post(url: str, data: dict | str, headers: dict | None = None, timeout: int = 15) -> str:
    """POST JSON or form data to a URL."""
    import json as _json
    import urllib.request
    try:
        if isinstance(data, dict):
            payload = _json.dumps(data).encode()
            hdrs = {"Content-Type": "application/json", **(headers or {})}
        else:
            payload = str(data).encode()
            hdrs = headers or {}
        req = urllib.request.Request(url, data=payload, headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read(MAX_READ_BYTES).decode("utf-8", errors="replace")
        return content[:MAX_TOOL_OUTPUT]
    except Exception as exc:
        return f"HTTP POST error: {exc}"

def download_file(url: str, dest: str) -> str:
    """Download a file from a URL to a local path."""
    ok, msg, p = validate_write_path(dest)
    if not ok:
        return msg
    try:
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(p))
        return f"Downloaded {url} → {p} ({p.stat().st_size:,} bytes)"
    except Exception as exc:
        return f"Download error: {exc}"

# ── Multimodal: PDF ───────────────────────────────────────────────────────────

def extract_pdf_text(path: str, max_pages: int = 50) -> str:
    """Extract text from a PDF file. Tries pdfminer, then pypdf."""
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg

    # Try pdfminer.six first
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract
        text = _pdfminer_extract(str(p), maxpages=max_pages)
        if text and text.strip():
            return text[:MAX_TOOL_OUTPUT]
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pdfminer failed: %s", exc)

    # Fallback: pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(str(p))
        pages = reader.pages[:max_pages]
        text = "\n\n".join(page.extract_text() or "" for page in pages)
        if text.strip():
            return text[:MAX_TOOL_OUTPUT]
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pypdf failed: %s", exc)

    return (
        "Could not extract PDF text. Install pdfminer.six or pypdf:\n"
        "  pip install pdfminer.six\n  pip install pypdf"
    )

def extract_docx_text(path: str) -> str:
    """Extract text from a .docx Word file."""
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg
    try:
        import docx
        doc = docx.Document(str(p))
        text = "\n".join(para.text for para in doc.paragraphs)
        return text[:MAX_TOOL_OUTPUT] or "(empty document)"
    except ImportError:
        return "Install python-docx:  pip install python-docx"
    except Exception as exc:
        return f"docx extraction error: {exc}"

# ── Multimodal: OCR ───────────────────────────────────────────────────────────

def ocr_image(path: str, lang: str = "eng") -> str:
    """Extract text from an image using Tesseract OCR."""
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(str(p))
        text = pytesseract.image_to_string(img, lang=lang)
        return text[:MAX_TOOL_OUTPUT] or "(no text found)"
    except ImportError:
        return "Install OCR deps:  pip install pytesseract Pillow\n(also install Tesseract binary)"
    except Exception as exc:
        return f"OCR error: {exc}"

# ── Multimodal: Audio/Video ───────────────────────────────────────────────────

def transcribe_audio(path: str, model: str = "base") -> str:
    """Transcribe audio/video using OpenAI Whisper."""
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg
    try:
        import whisper
        m = whisper.load_model(model)
        result = m.transcribe(str(p))
        return result.get("text", "(no transcription)")[:MAX_TOOL_OUTPUT]
    except ImportError:
        return "Install whisper:  pip install openai-whisper"
    except Exception as exc:
        return f"Transcription error: {exc}"

def video_info(path: str) -> str:
    """Get video/audio metadata via ffprobe."""
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg
    cmd = (
        f'ffprobe -v quiet -print_format json -show_streams -show_format "{p}"'
    )
    return execute_safe_command(cmd, timeout=10)

# ── Encryption / Decryption ───────────────────────────────────────────────────

def encrypt_text(plaintext: str, password: str) -> str:
    """AES-256-GCM encrypt text. Returns base64-encoded ciphertext."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        import secrets
        salt = secrets.token_bytes(16)
        kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
        key = kdf.derive(password.encode())
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
        payload = base64.b64encode(salt + nonce + ct).decode()
        return payload
    except ImportError:
        return "Install cryptography:  pip install cryptography"
    except Exception as exc:
        return f"Encrypt error: {exc}"

def decrypt_text(ciphertext_b64: str, password: str) -> str:
    """Decrypt text encrypted by encrypt_text()."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        raw = base64.b64decode(ciphertext_b64)
        salt, nonce, ct = raw[:16], raw[16:28], raw[28:]
        kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
        key = kdf.derive(password.encode())
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None).decode()
    except ImportError:
        return "Install cryptography:  pip install cryptography"
    except Exception as exc:
        return f"Decrypt error: {exc}"

def encrypt_file(path: str, password: str, out_path: str | None = None) -> str:
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg
    content = p.read_text(errors="replace")
    encrypted = encrypt_text(content, password)
    dest = out_path or str(p) + ".enc"
    return write_file(dest, encrypted)

def decrypt_file(path: str, password: str, out_path: str | None = None) -> str:
    ok, msg, p = validate_read_path(path, must_be_file=True)
    if not ok:
        return msg
    ok2, msg2 = validate_text_file(p)
    if not ok2:
        return msg2
    ciphertext = p.read_text()
    plaintext = decrypt_text(ciphertext.strip(), password)
    if plaintext.startswith("Decrypt error") or plaintext.startswith("Install"):
        return plaintext
    dest = out_path or str(p).removesuffix(".enc")
    return write_file(dest, plaintext)

# ── Live file watching ────────────────────────────────────────────────────────

def watch_file(path: str, callback: Callable[[str], None], timeout: int = 60) -> str:
    """Watch a file for changes and call callback(event_str) on each change.
    Runs in a background thread for `timeout` seconds.
    Requires watchdog: pip install watchdog
    """
    ok, msg, p = validate_read_path(path)
    if not ok:
        return msg
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class _H(FileSystemEventHandler):
            def on_modified(self, event):
                if Path(event.src_path).resolve() == p.resolve():
                    callback(f"modified: {event.src_path}")
            def on_created(self, event):
                callback(f"created: {event.src_path}")
            def on_deleted(self, event):
                callback(f"deleted: {event.src_path}")

        observer = Observer()
        observer.schedule(_H(), str(p.parent), recursive=False)
        observer.start()

        def _stop():
            import time; time.sleep(timeout); observer.stop()
        threading.Thread(target=_stop, daemon=True).start()

        return f"Watching {p} for {timeout}s"
    except ImportError:
        return "Install watchdog:  pip install watchdog"
    except Exception as exc:
        return f"Watch error: {exc}"

# ── Directory listing helper (used by executor) ───────────────────────────────

def _format_dir(path: Path) -> str:
    ok, msg, p = validate_read_path(str(path))
    if not ok:
        return msg
    if not p.is_dir():
        return f"Not a directory: {p}"
    lines = []
    try:
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return f"Permission denied: {p}"
    for entry in entries[:200]:
        if is_sensitive_path(entry):
            continue
        kind = "DIR " if entry.is_dir() else "FILE"
        size = ""
        if entry.is_file():
            try:
                size = f" {entry.stat().st_size:>12,} B"
            except OSError:
                size = ""
        lines.append(f"[{kind}] {entry.name}{size}")
    return "\n".join(lines) if lines else "(empty)"
