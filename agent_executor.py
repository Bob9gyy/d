"""
agent_executor.py — JARVIS tool-execution layer v2.1

New vs v2.0:
 - write_file, append_file, move_file, copy_file, delete_file
 - create_dir, delete_dir
 - read_any_file  (reads binary files as hex, text files as utf-8)
 - extract_pdf    (pdfminer / pypdf)
 - extract_docx   (python-docx)
 - ocr_image      (pytesseract + Pillow)
 - transcribe     (whisper)
 - video_info     (ffprobe)
 - http_get, http_post, download_file
 - encrypt_text, decrypt_text, encrypt_file, decrypt_file
 - pip_install
 - run_background
 - watch_file     (watchdog)
 - All tools surface through the same plan-execute loop
"""

from __future__ import annotations

import json
import logging
import re
from typing import Generator

from llm.llmrouter import LLMRouter
from jarvis_safety import (
    SAFE_TEXT_EXTENSIONS, MAX_TOOL_OUTPUT,
    # read
    validate_read_path, validate_text_file, safe_iter_files,
    execute_safe_command, open_safe_app, open_safe_url, search_web,
    # write / fs
    write_file, move_file, copy_file, delete_file, create_dir, delete_dir,
    # multimodal
    extract_pdf_text, extract_docx_text, ocr_image, transcribe_audio, video_info,
    # network
    http_get, http_post, download_file,
    # crypto
    encrypt_text, decrypt_text, encrypt_file, decrypt_file,
    # package / bg
    pip_install, run_background_task, watch_file,
    # dir format helper
    _format_dir,
)

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

AGENT_SYSTEM_SUFFIX = """
== TOOL PROTOCOL ==
When the user asks you to perform a task using a tool, respond with ONLY this JSON (no extra text):
{"action": "<tool_name>", "args": {<key>: <value>}}

Available tools:
  FILE READ:
    read_file(path)               — read any text/binary file
    read_any_file(path)           — read with hex fallback for binary
    list_dir(path)                — list folder contents
    search_files(query, root="~") — search files by name or content

  FILE WRITE:
    write_file(path, content)               — create/overwrite a file
    append_file(path, content)              — append to a file
    move_file(src, dst)                     — move/rename
    copy_file(src, dst)                     — copy
    delete_file(path, confirm_token=None)   — delete (asks for confirmation)
    create_dir(path)                        — make directory
    delete_dir(path, confirm_token=None)    — remove directory tree

  MULTIMODAL:
    extract_pdf(path, max_pages=50)   — extract text from PDF
    extract_docx(path)                — extract text from Word doc
    ocr_image(path, lang="eng")       — OCR text from image
    transcribe(path, model="base")    — speech-to-text (Whisper)
    video_info(path)                  — metadata via ffprobe

  NETWORK:
    http_get(url, headers=None)            — HTTP GET request
    http_post(url, data, headers=None)     — HTTP POST request
    download_file(url, dest)               — download to local path

  ENCRYPTION:
    encrypt_text(text, password)                   — AES-256-GCM encrypt
    decrypt_text(ciphertext, password)             — decrypt
    encrypt_file(path, password, out_path=None)    — encrypt a file
    decrypt_file(path, password, out_path=None)    — decrypt a file

  SYSTEM:
    system_info(metric="all")     — cpu/ram/disk/battery
    open_app(app)                 — launch an application
    open_url(url)                 — open browser URL
    search_web(query)             — open Google search
    run_command(command)          — run shell command
    pip_install(package)          — install Python package
    run_background(command,label) — run in background thread
    watch_file(path, timeout=60)  — watch file for changes

  MEMORY:
    index_directory(path)  — index folder into memory
    memory_search(query)   — search past conversations

After receiving a [TOOL RESULT], respond naturally in plain English.
Never show the JSON protocol to the user.
"""


class AgentExecutor:

    def __init__(self, llm: LLMRouter, memory=None) -> None:
        self.llm = llm
        self.memory = memory
        self._watch_log: list[str] = []

    def build_messages(
        self,
        base_system: str,
        conversation: list[dict],
        agent_mode: bool = False,
    ) -> list[dict]:
        sys_content = base_system + (AGENT_SYSTEM_SUFFIX if agent_mode else "")
        return [{"role": "system", "content": sys_content}] + list(conversation)

    def run(
        self,
        messages: list[dict],
        model: str | None = None,
        agent_mode: bool = False,
        max_tool_rounds: int = 10,
    ) -> Generator[str, None, None]:
        if not agent_mode:
            yield from self.llm.chat(messages, model=model)
            return

        msgs = list(messages)
        for _ in range(max_tool_rounds):
            raw = ""
            for tok in self.llm.chat(msgs, model=model, stream=True):
                raw += tok
                if not raw.lstrip().startswith("{"):
                    yield tok

            action, args = self._parse_action(raw)
            if action is None:
                if raw.lstrip().startswith("{"):
                    yield raw
                return

            yield f"\n⚙ [{action}] …\n"
            result = self._execute(action, args)
            yield f"```\n{result}\n```\n"

            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "user", "content": f"[TOOL RESULT]\n{result}"})

        yield "\n⚠️  Agent reached max tool rounds.\n"

    # ── Action parsing ────────────────────────────────────────────────────────

    def _parse_action(self, text: str) -> tuple[str | None, dict]:
        stripped = re.sub(r"^```(?:json)?\s*", "", text.strip())
        stripped = re.sub(r"\s*```$", "", stripped)
        if not stripped.startswith("{"):
            return None, {}
        try:
            data = json.loads(stripped)
            action = data.get("action", "")
            args   = data.get("args", {})
            if isinstance(action, str) and action and isinstance(args, dict):
                return action, args
        except (json.JSONDecodeError, ValueError):
            pass
        return None, {}

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _execute(self, action: str, args: dict) -> str:
        dispatch = {
            # read
            "read_file":       self._tool_read_file,
            "read_any_file":   self._tool_read_any_file,
            "list_dir":        self._tool_list_dir,
            "search_files":    self._tool_search_files,
            # write
            "write_file":      self._tool_write_file,
            "append_file":     self._tool_append_file,
            "move_file":       self._tool_move_file,
            "copy_file":       self._tool_copy_file,
            "delete_file":     self._tool_delete_file,
            "create_dir":      self._tool_create_dir,
            "delete_dir":      self._tool_delete_dir,
            # multimodal
            "extract_pdf":     self._tool_extract_pdf,
            "extract_docx":    self._tool_extract_docx,
            "ocr_image":       self._tool_ocr_image,
            "transcribe":      self._tool_transcribe,
            "video_info":      self._tool_video_info,
            # network
            "http_get":        self._tool_http_get,
            "http_post":       self._tool_http_post,
            "download_file":   self._tool_download_file,
            # crypto
            "encrypt_text":    self._tool_encrypt_text,
            "decrypt_text":    self._tool_decrypt_text,
            "encrypt_file":    self._tool_encrypt_file,
            "decrypt_file":    self._tool_decrypt_file,
            # system
            "system_info":     self._tool_system_info,
            "open_app":        self._tool_open_app,
            "open_url":        self._tool_open_url,
            "search_web":      self._tool_search_web,
            "run_command":     self._tool_run_command,
            "pip_install":     self._tool_pip_install,
            "run_background":  self._tool_run_background,
            "watch_file":      self._tool_watch_file,
            # memory
            "index_directory": self._tool_index_directory,
            "memory_search":   self._tool_memory_search,
        }
        fn = dispatch.get(action)
        if fn is None:
            return f"Unknown tool: {action}"
        try:
            return fn(**args)
        except TypeError as exc:
            return f"Tool argument error for '{action}': {exc}"
        except Exception as exc:
            logger.exception("Tool error: %s", action)
            return f"Tool error: {exc}"

    # ── Read tools ────────────────────────────────────────────────────────────

    def _tool_read_file(self, path: str) -> str:
        ok, msg, p = validate_read_path(path, must_be_file=True)
        if not ok:
            return msg
        ok2, msg2 = validate_text_file(p)
        if not ok2:
            return msg2
        try:
            text = p.read_text(errors="replace")
            tail = "\n… [truncated]" if len(text) > MAX_TOOL_OUTPUT else ""
            return text[:MAX_TOOL_OUTPUT] + tail
        except PermissionError:
            return f"Permission denied: {p}"

    def _tool_read_any_file(self, path: str) -> str:
        """Read any file; falls back to hex dump for binary."""
        ok, msg, p = validate_read_path(path, must_be_file=True)
        if not ok:
            return msg
        ok2, msg2 = validate_text_file(p)
        if not ok2:
            return msg2
        try:
            raw = p.read_bytes()
        except PermissionError:
            return f"Permission denied: {p}"
        # Try utf-8
        try:
            text = raw.decode("utf-8")
            tail = "\n… [truncated]" if len(text) > MAX_TOOL_OUTPUT else ""
            return text[:MAX_TOOL_OUTPUT] + tail
        except UnicodeDecodeError:
            pass
        # Hex dump
        lines = []
        for i in range(0, min(len(raw), 4096), 16):
            chunk = raw[i:i+16]
            hex_part  = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{i:08x}  {hex_part:<47}  {ascii_part}")
        note = f"[Binary file, showing first {min(len(raw),4096)} bytes as hex]\n"
        return note + "\n".join(lines)

    def _tool_list_dir(self, path: str = "~") -> str:
        ok, msg, p = validate_read_path(path)
        if not ok:
            return msg
        if not p.is_dir():
            return f"Not a directory: {p}"
        return f"Contents of {p}:\n" + _format_dir(p)

    def _tool_search_files(self, query: str, root: str = "~") -> str:
        ok, msg, root_p = validate_read_path(root)
        if not ok:
            return msg
        if not root_p.is_dir():
            return f"Not a directory: {root_p}"
        query_low = query.lower()
        hits: list[str] = []
        for p in safe_iter_files(root_p, max_scan=20_000):
            if query_low in p.name.lower():
                hits.append(f"[name match] {p}")
            elif p.suffix.lower() in SAFE_TEXT_EXTENSIONS:
                try:
                    ok2, _ = validate_text_file(p)
                    if not ok2:
                        continue
                    content = p.read_text(errors="replace")
                    if query_low in content.lower():
                        for ln in content.splitlines():
                            if query_low in ln.lower():
                                hits.append(f"[content] {p}\n ↳ {ln.strip()[:120]}")
                                break
                except Exception:
                    continue
            if len(hits) >= 50:
                break
        if not hits:
            return f"No files found matching '{query}' under {root_p}"
        return f"Found {len(hits)} result(s) for '{query}':\n" + "\n".join(hits)

    # ── Write tools ───────────────────────────────────────────────────────────

    def _tool_write_file(self, path: str, content: str) -> str:
        return write_file(path, content, append=False)

    def _tool_append_file(self, path: str, content: str) -> str:
        return write_file(path, content, append=True)

    def _tool_move_file(self, src: str, dst: str, confirm_token: str | None = None) -> str:
        return move_file(src, dst, confirm_token=confirm_token)

    def _tool_copy_file(self, src: str, dst: str) -> str:
        return copy_file(src, dst)

    def _tool_delete_file(self, path: str, confirm_token: str | None = None) -> str:
        return delete_file(path, confirm_token=confirm_token)

    def _tool_create_dir(self, path: str) -> str:
        return create_dir(path)

    def _tool_delete_dir(self, path: str, confirm_token: str | None = None) -> str:
        return delete_dir(path, confirm_token=confirm_token)

    # ── Multimodal tools ──────────────────────────────────────────────────────

    def _tool_extract_pdf(self, path: str, max_pages: int = 50) -> str:
        return extract_pdf_text(path, max_pages=max_pages)

    def _tool_extract_docx(self, path: str) -> str:
        return extract_docx_text(path)

    def _tool_ocr_image(self, path: str, lang: str = "eng") -> str:
        return ocr_image(path, lang=lang)

    def _tool_transcribe(self, path: str, model: str = "base") -> str:
        return transcribe_audio(path, model=model)

    def _tool_video_info(self, path: str) -> str:
        return video_info(path)

    # ── Network tools ─────────────────────────────────────────────────────────

    def _tool_http_get(self, url: str, headers: dict | None = None) -> str:
        return http_get(url, headers=headers)

    def _tool_http_post(self, url: str, data: dict | str, headers: dict | None = None) -> str:
        return http_post(url, data, headers=headers)

    def _tool_download_file(self, url: str, dest: str) -> str:
        return download_file(url, dest)

    # ── Crypto tools ──────────────────────────────────────────────────────────

    def _tool_encrypt_text(self, text: str, password: str) -> str:
        return encrypt_text(text, password)

    def _tool_decrypt_text(self, ciphertext: str, password: str) -> str:
        return decrypt_text(ciphertext, password)

    def _tool_encrypt_file(self, path: str, password: str, out_path: str | None = None) -> str:
        return encrypt_file(path, password, out_path=out_path)

    def _tool_decrypt_file(self, path: str, password: str, out_path: str | None = None) -> str:
        return decrypt_file(path, password, out_path=out_path)

    # ── System tools ──────────────────────────────────────────────────────────

    def _tool_system_info(self, metric: str = "all") -> str:
        if not _PSUTIL:
            return "psutil not installed. Run: pip install psutil"
        parts = []
        m = metric.lower()
        if m in ("cpu", "all"):
            parts.append(f"CPU : {psutil.cpu_percent(interval=0.5):.1f}% | {psutil.cpu_count()} cores")
        if m in ("ram", "memory", "all"):
            vm = psutil.virtual_memory()
            parts.append(f"RAM : {vm.percent:.1f}% | {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB")
        if m in ("disk", "all"):
            for part in psutil.disk_partitions():
                try:
                    u = psutil.disk_usage(part.mountpoint)
                    parts.append(f"DISK [{part.mountpoint}]: {u.percent:.1f}% | {u.used/1e9:.1f}/{u.total/1e9:.1f} GB")
                except Exception:
                    pass
        if m in ("battery", "all"):
            bat = psutil.sensors_battery()
            if bat:
                parts.append(f"BATT : {bat.percent:.0f}%{' [CHARGING]' if bat.power_plugged else ''}")
        return "\n".join(parts) if parts else f"Unknown metric: {metric}"

    def _tool_open_app(self, app: str) -> str:
        return open_safe_app(app)

    def _tool_open_url(self, url: str) -> str:
        return open_safe_url(url)

    def _tool_search_web(self, query: str) -> str:
        return search_web(query)

    def _tool_run_command(self, command: str, cwd: str | None = None, timeout: int = 30) -> str:
        return execute_safe_command(command, cwd=cwd, timeout=timeout)

    def _tool_pip_install(self, package: str) -> str:
        return pip_install(package)

    def _tool_run_background(self, command: str, label: str = "task") -> str:
        return run_background_task(command, label=label)

    def _tool_watch_file(self, path: str, timeout: int = 60) -> str:
        def _cb(event: str):
            self._watch_log.append(event)
        return watch_file(path, callback=_cb, timeout=timeout)

    # ── Memory tools ──────────────────────────────────────────────────────────

    def _tool_index_directory(self, path: str) -> str:
        if self.memory is None:
            return "Memory engine not attached."
        ok, msg, p = validate_read_path(path)
        if not ok:
            return msg
        try:
            from file_indexer import FileIndexer
            count = FileIndexer(self.memory).index_directory(str(p))
            return f"Indexed {count} chunk(s) from {p}"
        except ImportError:
            count = sum(
                1 for fp in safe_iter_files(p)
                if fp.suffix.lower() in SAFE_TEXT_EXTENSIONS
                and self.memory.index_file(str(fp))
            )
            return f"Indexed {count} file(s) from {p}"

    def _tool_memory_search(self, query: str) -> str:
        if self.memory is None:
            return "Memory engine not attached."
        results = self.memory.retrieve(query, n_results=5)
        return ("\n---\n".join(results[:5])) if results else f"No memories found for: {query}"
