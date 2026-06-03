"""
jarvis_safety.py - conservative safety helpers for local desktop tools.

This module centralizes policy for actions requested by the LLM. The goal is
not to make arbitrary OS automation safe; it is to keep the assistant useful
while defaulting to read-only, local-first behavior.
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import re
import shlex
import subprocess
import webbrowser
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus, urlparse


MAX_READ_BYTES = 512_000
MAX_TOOL_OUTPUT = 4_000

SAFE_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rb",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".tsv",
    ".html", ".htm", ".xml",
}

SKIP_DIRS = {
    "node_modules", ".git", ".svn", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".idea", ".vscode", "coverage",
    "$recycle.bin", "windows", "program files", "program files (x86)",
    "appdata", "programdata",
}

SENSITIVE_NAMES = {
    ".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_dsa",
    "id_ecdsa", "id_ed25519", "known_hosts", "credentials",
    "credentials.json", "token", "tokens.json",
}

SENSITIVE_SUFFIXES = {
    ".key", ".pem", ".pfx", ".p12", ".keystore", ".kdbx", ".sqlite",
    ".db", ".db3",
}

SHELL_METACHARS = re.compile(r"[|&;<>()`$]")
DANGEROUS_COMMANDS = {
    "rm", "del", "erase", "rmdir", "rd", "remove-item", "move-item",
    "copy-item", "format", "shutdown", "restart-computer", "poweroff",
    "reboot", "mkfs", "diskpart", "reg", "sudo", "su", "chmod", "chown",
}

READ_ONLY_EXTERNAL_COMMANDS = {
    "whoami", "hostname", "ipconfig", "ifconfig", "ping", "tracert",
    "traceroute", "netstat", "tasklist", "ps", "df", "du", "free",
    "uname",
}

READ_ONLY_GIT_SUBCOMMANDS = {
    "status", "log", "diff", "show", "rev-parse", "branch",
}

BLOCKED_APPS = {"cmd", "powershell", "terminal", "bash", "sh", "zsh"}

APP_COMMANDS_WINDOWS = {
    "chrome": ["chrome"],
    "firefox": ["firefox"],
    "brave": ["brave"],
    "edge": ["msedge"],
    "notepad": ["notepad"],
    "calculator": ["calc"],
    "explorer": ["explorer"],
    "word": ["winword"],
    "excel": ["excel"],
    "vscode": ["code"],
    "paint": ["mspaint"],
    "spotify": ["spotify"],
}

APP_COMMANDS_POSIX = {
    "chrome": ["google-chrome"],
    "firefox": ["firefox"],
    "brave": ["brave-browser"],
    "vscode": ["code"],
    "calculator": ["gnome-calculator"],
    "spotify": ["spotify"],
    "notepad": ["gedit"],
    "explorer": ["nautilus"],
}


def allowed_roots() -> list[Path]:
    """Return directories Jarvis may read/index by default."""
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
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        bool(parts & SKIP_DIRS)
        or name in SENSITIVE_NAMES
        or path.suffix.lower() in SENSITIVE_SUFFIXES
    )


def validate_read_path(path: str | None, *, must_be_file: bool = False) -> tuple[bool, str, Path]:
    p = resolve_path(path)
    if not is_under(p, allowed_roots()):
        return False, f"Blocked path outside allowed roots: {p}", p
    if is_sensitive_path(p):
        return False, f"Blocked sensitive path: {p}", p
    if not p.exists():
        return False, f"Path not found: {p}", p
    if must_be_file and not p.is_file():
        return False, f"Not a file: {p}", p
    return True, "", p


def validate_text_file(path: Path) -> tuple[bool, str]:
    if path.suffix.lower() not in SAFE_TEXT_EXTENSIONS:
        return False, f"Unsupported file type for safe text read: {path.suffix or '(none)'}"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"Cannot stat file: {exc}"
    if size > MAX_READ_BYTES:
        return False, f"File is too large for safe read: {size:,} bytes"
    return True, ""


def safe_iter_files(root: Path, max_scan: int = 5_000):
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


def _split_command(command: str) -> tuple[bool, str, list[str]]:
    if not command or not command.strip():
        return False, "Empty command.", []
    if SHELL_METACHARS.search(command):
        return False, "Blocked shell metacharacters. Use simple read-only commands only.", []
    try:
        parts = shlex.split(command, posix=(platform.system() != "Windows"))
    except ValueError as exc:
        return False, f"Could not parse command: {exc}", []
    if not parts:
        return False, "Empty command.", []
    normalized = [p.strip("\"'") for p in parts]
    base = Path(normalized[0]).name.lower().removesuffix(".exe")
    if base in DANGEROUS_COMMANDS:
        return False, f"Blocked dangerous command: {base}", []
    return True, "", normalized


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
    for entry in entries[:80]:
        if is_sensitive_path(entry):
            continue
        kind = "DIR " if entry.is_dir() else "FILE"
        size = ""
        if entry.is_file():
            try:
                size = f"  {entry.stat().st_size:>10,} B"
            except OSError:
                size = ""
        lines.append(f"[{kind}] {entry.name}{size}")
    return "\n".join(lines) if lines else "(empty)"


def execute_safe_command(command: str, cwd: str | None = None) -> str:
    ok, msg, parts = _split_command(command)
    if not ok:
        return msg

    base = Path(parts[0]).name.lower().removesuffix(".exe")
    run_cwd = resolve_path(cwd, default=".") if cwd else Path.cwd().resolve()
    if not is_under(run_cwd, allowed_roots()):
        return f"Blocked working directory outside allowed roots: {run_cwd}"

    if base in {"pwd", "cd"}:
        return str(run_cwd)
    if base in {"date", "time"}:
        return _dt.datetime.now().isoformat(timespec="seconds")
    if base in {"dir", "ls"}:
        target = resolve_path(parts[1], default=str(run_cwd)) if len(parts) > 1 else run_cwd
        return _format_dir(target)[:MAX_TOOL_OUTPUT]
    if base == "echo":
        return " ".join(parts[1:])[:MAX_TOOL_OUTPUT] if len(parts) > 1 else ""
    if base == "ver":
        return platform.platform()
    if base == "git":
        if len(parts) < 2 or parts[1].lower() not in READ_ONLY_GIT_SUBCOMMANDS:
            return "Blocked git command. Allowed subcommands: " + ", ".join(sorted(READ_ONLY_GIT_SUBCOMMANDS))
    elif base not in READ_ONLY_EXTERNAL_COMMANDS:
        allowed = sorted(READ_ONLY_EXTERNAL_COMMANDS | {"date", "dir", "echo", "git", "ls", "pwd", "time", "ver"})
        return f"Command '{base}' is not allowed. Allowed read-only commands: {', '.join(allowed)}"

    try:
        result = subprocess.run(
            parts,
            shell=False,
            cwd=str(run_cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return f"Command not found: {parts[0]}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 10 seconds."
    except Exception as exc:
        return f"Error running command: {exc}"

    output = result.stdout.strip() or result.stderr.strip() or "(no output)"
    if result.returncode != 0:
        output = f"[exit {result.returncode}]\n{output}"
    return output[:MAX_TOOL_OUTPUT]


def open_safe_app(app: str) -> str:
    app_name = (app or "").lower().strip()
    if not app_name:
        return "No app name provided."
    if app_name in BLOCKED_APPS:
        return f"Blocked launching shell app '{app_name}' for safety."

    commands = APP_COMMANDS_WINDOWS if platform.system() == "Windows" else APP_COMMANDS_POSIX
    cmd = commands.get(app_name)
    if not cmd:
        return "App not in safe launch list: " + app_name

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
        return "Blocked URL. Only http(s) URLs with a hostname are allowed."
    webbrowser.open(candidate)
    return f"Opened URL: {candidate}"


def search_web(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "No search query provided."
    url = "https://www.google.com/search?q=" + quote_plus(q)
    webbrowser.open(url)
    return f"Searched the web for: {q}"
