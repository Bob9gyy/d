"""
main_ui.py — Jarvis main chat window
Dark HUD-style chat UI using PySide6. Supports streaming LLM responses,
persistent memory, live system stats, and system tray integration.

FIXES APPLIED:
  - [FIX-1] _on_token: added processEvents() → streaming no longer lags
  - [FIX-2] _send: resolved model BEFORE starting thread → no race condition
  - [FIX-3] _update_stats: Ollama health check moved to background QThread
  - [FIX-4] psutil.cpu_percent primed at startup → no false 0% on first read
  - [FIX-5] _quit(): clean shutdown (stops timers, joins worker) wired to tray
  - [FIX-9] _save_transcript: saves to ~/jarvis_transcripts/ not random CWD
"""
import json
import datetime
import pathlib

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QComboBox,
    QFrame, QSystemTrayIcon, QMenu, QApplication,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QObject, QRunnable, QThreadPool
from PySide6.QtGui import (
    QFont, QColor, QIcon, QTextCursor, QTextCharFormat,
    QTextBlockFormat, QPainter, QPixmap, QKeyEvent,
)

try:
    import psutil
    _PSUTIL = True
    psutil.cpu_percent(interval=None)   # FIX-4: prime the counter so first real read is valid
except ImportError:
    _PSUTIL = False

from llm_router import LLMRouter
from memory import MemoryEngine

# ── Constants ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are JARVIS (Just A Rather Very Intelligent System), a local offline AI assistant.\n"
    "You run entirely on the user's machine — no cloud, no telemetry, no data leaves the device.\n"
    "Tone: calm, precise, professional, with occasional dry wit. Like a brilliant colleague.\n"
    "Be concise by default. Use code blocks where helpful. Avoid unnecessary filler."
)

COL_BG       = "#020c10"
COL_SIDEBAR  = "#010a0d"
COL_BORDER   = "#002838"
COL_INPUT    = "#011520"
COL_CYAN     = "#00d4ff"
COL_DIM      = "#006080"
COL_GREEN    = "#00ffb4"
COL_TEXT     = "#cde8f0"
COL_LABEL_U  = "#00ffb4"   # user label
COL_LABEL_J  = "#00d4ff"   # jarvis label

# ── LLM stream worker ──────────────────────────────────────────────────────────

class StreamWorker(QThread):
    token  = Signal(str)
    done   = Signal(str)   # full assembled response

    def __init__(self, llm: LLMRouter, messages: list, model: str):
        super().__init__()
        self.llm      = llm
        self.messages = messages
        self.model    = model   # FIX-2: model resolved before thread starts
        self._text    = ""

    def run(self):
        try:
            for tok in self.llm.chat(self.messages, model=self.model):
                self._text += tok
                self.token.emit(tok)
        except Exception as exc:
            err = f"\n\n⚠️  Unexpected error: {exc}"
            self._text += err
            self.token.emit(err)
        self.done.emit(self._text)


# ── FIX-3: Non-blocking Ollama health check ────────────────────────────────────

class _OllamaCheckSignals(QObject):
    result = Signal(bool)

class _OllamaCheckTask(QRunnable):
    """Checks Ollama reachability in the thread pool — never blocks the UI."""
    def __init__(self, llm: LLMRouter, signals: _OllamaCheckSignals):
        super().__init__()
        self.llm     = llm
        self.signals = signals

    def run(self):
        online = self.llm.is_online()
        self.signals.result.emit(online)


# ── Stylesheet ─────────────────────────────────────────────────────────────────

APP_STYLE = f"""
QMainWindow, QWidget#root, QWidget#sidebar, QWidget#chat_panel {{
    background-color: {COL_BG};
}}
QWidget#sidebar {{
    background-color: {COL_SIDEBAR};
    border-right: 1px solid {COL_BORDER};
}}
QTextEdit#chat_log {{
    background-color: {COL_BG};
    color: {COL_TEXT};
    border: none;
    font-family: Consolas;
    font-size: 10pt;
    selection-background-color: #003850;
    padding: 8px;
}}
QTextEdit#input_field {{
    background-color: {COL_INPUT};
    color: {COL_TEXT};
    border: 1px solid #003850;
    border-radius: 4px;
    font-family: Consolas;
    font-size: 10pt;
    padding: 6px 10px;
}}
QTextEdit#input_field:focus {{
    border: 1px solid {COL_CYAN};
}}
QPushButton#send_btn {{
    background-color: #003d55;
    color: {COL_CYAN};
    border: 1px solid #005870;
    border-radius: 4px;
    font-family: Consolas;
    font-weight: bold;
    font-size: 10pt;
    padding: 8px 18px;
    min-width: 74px;
}}
QPushButton#send_btn:hover  {{ background-color: #00516e; border-color: #00a0c0; }}
QPushButton#send_btn:pressed {{ background-color: #002a40; }}
QPushButton#send_btn:disabled {{ background-color: #001a24; color: #004050; border-color: #002030; }}
QLabel {{ color: {COL_CYAN}; font-family: Consolas; }}
QComboBox {{
    background-color: {COL_INPUT};
    color: {COL_CYAN};
    border: 1px solid {COL_BORDER};
    border-radius: 3px;
    font-family: Consolas;
    font-size: 9pt;
    padding: 3px 8px;
}}
QComboBox QAbstractItemView {{
    background-color: #010f18;
    color: {COL_CYAN};
    border: 1px solid {COL_BORDER};
    selection-background-color: #003050;
}}
QComboBox::drop-down {{ border: none; }}
QScrollBar:vertical {{
    background: {COL_SIDEBAR}; width: 7px; border: none;
}}
QScrollBar::handle:vertical {{
    background: #003040; border-radius: 3px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: #005060; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QPushButton.tool_btn {{
    background-color: {COL_INPUT};
    color: #00a0b8;
    border: 1px solid #002838;
    border-radius: 3px;
    font-family: Consolas;
    font-size: 8pt;
    padding: 4px 8px;
    text-align: left;
}}
QPushButton.tool_btn:hover {{ background-color: #012030; color: {COL_CYAN}; }}
"""


class JarvisWindow(QMainWindow):
    def __init__(self, llm: LLMRouter, memory: MemoryEngine):
        super().__init__()
        self.llm            = llm
        self.memory         = memory
        self.conversation   : list[dict] = []
        self.worker         : StreamWorker | None = None
        self._stream_active = False
        self._mem_enabled   = True
        self._auto_route    = True
        self._stream_fmt    : QTextCharFormat | None = None

        # FIX-3: thread pool for async health checks
        self._pool          = QThreadPool.globalInstance()
        self._check_signals = _OllamaCheckSignals()
        self._check_signals.result.connect(self._apply_online_status)

        self.setWindowTitle("JARVIS  —  Local AI")
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(APP_STYLE)

        self._build_ui()
        self._build_tray()
        self._poll_stats()   # immediate first poll (async)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._poll_stats)
        self._stats_timer.start(4000)

        QTimer.singleShot(400, self._welcome)

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar(), 0)
        layout.addWidget(self._build_chat_panel(), 1)

    # ── Sidebar helpers ────────────────────────────────────────────────────────

    def _sb_section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Consolas", 8, QFont.Bold))
        lbl.setStyleSheet("color: #007a90; margin-top: 12px; margin-bottom: 2px; letter-spacing: 1px;")
        return lbl

    def _sb_info(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Consolas", 8))
        lbl.setStyleSheet("color: #005060; margin-left: 2px;")
        return lbl

    def _sb_btn(self, label: str, cb) -> QPushButton:
        btn = QPushButton(label)
        btn.setProperty("class", "tool_btn")
        btn.setFont(QFont("Consolas", 8))
        btn.clicked.connect(cb)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COL_INPUT}; color: #00a0b8;
                border: 1px solid #002838; border-radius: 3px;
                font-family: Consolas; font-size: 8pt;
                padding: 4px 8px; text-align: left;
            }}
            QPushButton:hover {{ background-color: #012030; color: {COL_CYAN}; }}
        """)
        return btn

    def _build_sidebar(self) -> QWidget:
        sb = QWidget()
        sb.setObjectName("sidebar")
        sb.setFixedWidth(234)
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(12, 14, 12, 12)
        lay.setSpacing(3)

        # Logo
        title = QLabel("J.A.R.V.I.S")
        title.setFont(QFont("Consolas", 13, QFont.Bold))
        title.setStyleSheet(f"color: {COL_CYAN}; letter-spacing: 3px;")
        lay.addWidget(title)

        ver = QLabel("Local AI  ·  v1.0  ·  Offline")
        ver.setFont(QFont("Consolas", 7))
        ver.setStyleSheet("color: #003850; margin-bottom: 6px;")
        lay.addWidget(ver)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {COL_BORDER}; border: none; max-height: 1px;")
        lay.addWidget(sep)

        # Model
        lay.addWidget(self._sb_section("MODEL"))
        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Consolas", 9))
        self._refresh_models()
        lay.addWidget(self.model_combo)

        self.auto_btn = self._sb_btn("⚡  Auto-Route:  ON", self._toggle_auto_route)
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(True)
        lay.addWidget(self.auto_btn)

        # Memory
        lay.addWidget(self._sb_section("MEMORY"))
        self.mem_btn = self._sb_btn("●  Memory:  ON", self._toggle_memory)
        self.mem_btn.setCheckable(True)
        self.mem_btn.setChecked(True)
        lay.addWidget(self.mem_btn)

        self.mem_count = self._sb_info(f"  {self.memory.count()} stored memories")
        lay.addWidget(self.mem_count)

        lay.addWidget(self._sb_btn("⊘  Clear All Memories", self._clear_memory))

        # System
        lay.addWidget(self._sb_section("SYSTEM"))
        self.cpu_lbl  = self._sb_info("  CPU  : —")
        self.ram_lbl  = self._sb_info("  RAM  : —")
        self.llm_lbl  = self._sb_info("  LLM  : checking …")
        lay.addWidget(self.cpu_lbl)
        lay.addWidget(self.ram_lbl)
        lay.addWidget(self.llm_lbl)

        # Tools
        lay.addWidget(self._sb_section("TOOLS"))
        lay.addWidget(self._sb_btn("⌖  New Chat", self._new_chat))
        lay.addWidget(self._sb_btn("⎙  Save Transcript", self._save_transcript))
        lay.addWidget(self._sb_btn("↺  Refresh Models", self._refresh_models))

        lay.addStretch()

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"background: {COL_BORDER}; border: none; max-height: 1px;")
        lay.addWidget(sep2)

        self.status_lbl = QLabel("● Connecting …")
        self.status_lbl.setFont(QFont("Consolas", 8))
        self.status_lbl.setStyleSheet("color: #444; margin-top: 6px;")
        lay.addWidget(self.status_lbl)

        return sb

    # ── Chat panel ─────────────────────────────────────────────────────────────

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("chat_panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.chat_log = QTextEdit()
        self.chat_log.setObjectName("chat_log")
        self.chat_log.setReadOnly(True)
        self.chat_log.setFont(QFont("Consolas", 10))
        lay.addWidget(self.chat_log, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {COL_BORDER}; border: none; max-height: 1px;")
        lay.addWidget(sep)

        bar = QWidget()
        bar.setStyleSheet(f"background-color: {COL_INPUT}; border-top: 1px solid {COL_BORDER};")
        b_lay = QHBoxLayout(bar)
        b_lay.setContentsMargins(12, 10, 12, 10)
        b_lay.setSpacing(8)

        self.input_field = QTextEdit()
        self.input_field.setObjectName("input_field")
        self.input_field.setPlaceholderText("Message JARVIS …  (Enter to send, Shift+Enter for newline)")
        self.input_field.setMaximumHeight(78)
        self.input_field.setMinimumHeight(42)
        self.input_field.installEventFilter(self)
        b_lay.addWidget(self.input_field, 1)

        self.send_btn = QPushButton("SEND")
        self.send_btn.setObjectName("send_btn")
        self.send_btn.clicked.connect(self._send)
        b_lay.addWidget(self.send_btn)

        lay.addWidget(bar)
        return panel

    # ── Event filter (Enter to send) ───────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.input_field and isinstance(event, QKeyEvent):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if not (event.modifiers() & Qt.ShiftModifier):
                    self._send()
                    return True
        return super().eventFilter(obj, event)

    # ── Chat logic ─────────────────────────────────────────────────────────────

    def _welcome(self):
        self._insert_system_msg(
            "J.A.R.V.I.S  ONLINE  —  ALL PROCESSING LOCAL  —  NO DATA LEAVES YOUR MACHINE\n"
            "Type a message below, or press Enter to send.  Shift+Enter = new line."
        )

    def _send(self):
        text = self.input_field.toPlainText().strip()
        if not text or self._stream_active:
            return

        self.input_field.clear()
        self.send_btn.setEnabled(False)
        self._stream_active = True

        self._insert_user_msg(text)
        self.conversation.append({"role": "user", "content": text})

        # Build message list for LLM
        sys_p = SYSTEM_PROMPT
        if self._mem_enabled:
            sys_p = self.memory.inject_context(text, sys_p)
        msgs = [{"role": "system", "content": sys_p}] + self.conversation[:]

        # FIX-2: resolve model NOW on the main thread before the worker starts,
        # so we never mutate shared LLMRouter state from a background thread.
        if self._auto_route:
            model = self.llm.route(text)
        else:
            model = self.model_combo.currentText() or self.llm.default_model

        self._begin_jarvis_block()

        self.worker = StreamWorker(self.llm, msgs, model=model)
        self.worker.token.connect(self._on_token)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    # ── Text insertion helpers ─────────────────────────────────────────────────

    def _cursor_at_end(self) -> QTextCursor:
        cur = self.chat_log.textCursor()
        cur.movePosition(QTextCursor.End)
        return cur

    def _insert_system_msg(self, text: str):
        cur = self._cursor_at_end()
        blk_fmt = QTextBlockFormat()
        blk_fmt.setTopMargin(10)
        blk_fmt.setBottomMargin(6)
        blk_fmt.setAlignment(Qt.AlignCenter)
        cur.insertBlock(blk_fmt)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#003d50"))
        fmt.setFont(QFont("Consolas", 8))
        cur.insertText(text, fmt)

        self.chat_log.setTextCursor(cur)
        self.chat_log.ensureCursorVisible()

    def _insert_user_msg(self, text: str):
        cur = self._cursor_at_end()

        blk_fmt = QTextBlockFormat()
        blk_fmt.setTopMargin(14)
        blk_fmt.setLeftMargin(80)
        blk_fmt.setRightMargin(12)
        cur.insertBlock(blk_fmt)

        lbl_fmt = QTextCharFormat()
        lbl_fmt.setForeground(QColor(COL_LABEL_U))
        lbl_fmt.setFont(QFont("Consolas", 8, QFont.Bold))
        cur.insertText("YOU", lbl_fmt)

        blk2 = QTextBlockFormat()
        blk2.setTopMargin(2)
        blk2.setLeftMargin(80)
        blk2.setRightMargin(12)
        blk2.setBottomMargin(4)
        cur.insertBlock(blk2)

        txt_fmt = QTextCharFormat()
        txt_fmt.setForeground(QColor(COL_TEXT))
        txt_fmt.setFont(QFont("Consolas", 10))
        cur.insertText(text, txt_fmt)

        self.chat_log.setTextCursor(cur)
        self.chat_log.ensureCursorVisible()

    def _begin_jarvis_block(self):
        cur = self._cursor_at_end()

        blk_fmt = QTextBlockFormat()
        blk_fmt.setTopMargin(10)
        blk_fmt.setLeftMargin(12)
        blk_fmt.setRightMargin(80)
        cur.insertBlock(blk_fmt)

        lbl_fmt = QTextCharFormat()
        lbl_fmt.setForeground(QColor(COL_LABEL_J))
        lbl_fmt.setFont(QFont("Consolas", 8, QFont.Bold))
        cur.insertText("◆ JARVIS", lbl_fmt)

        blk2 = QTextBlockFormat()
        blk2.setTopMargin(2)
        blk2.setLeftMargin(12)
        blk2.setRightMargin(80)
        blk2.setBottomMargin(4)
        cur.insertBlock(blk2)

        self._stream_fmt = QTextCharFormat()
        self._stream_fmt.setForeground(QColor(COL_TEXT))
        self._stream_fmt.setFont(QFont("Consolas", 10))

        self.chat_log.setTextCursor(cur)

    def _on_token(self, tok: str):
        # FIX-1: processEvents() forces Qt to repaint each token immediately
        # instead of batching repaints until the next mouse/keyboard event.
        cur = self._cursor_at_end()
        cur.insertText(tok, self._stream_fmt)
        self.chat_log.setTextCursor(cur)
        self.chat_log.ensureCursorVisible()
        QApplication.processEvents()

    def _on_done(self, full: str):
        cur = self._cursor_at_end()
        cur.insertBlock()
        self.chat_log.setTextCursor(cur)

        self.conversation.append({"role": "assistant", "content": full})

        if self._mem_enabled and len(self.conversation) >= 2:
            u = self.conversation[-2].get("content", "")
            self.memory.store(u, full)
            self.mem_count.setText(f"  {self.memory.count()} stored memories")

        self.send_btn.setEnabled(True)
        self._stream_active = False
        self.worker = None

    # ── Sidebar actions ────────────────────────────────────────────────────────

    def _toggle_memory(self):
        self._mem_enabled = self.mem_btn.isChecked()
        s = "ON" if self._mem_enabled else "OFF"
        self.mem_btn.setText(f"{'●' if self._mem_enabled else '○'}  Memory:  {s}")

    def _toggle_auto_route(self):
        self._auto_route = self.auto_btn.isChecked()
        s = "ON" if self._auto_route else "OFF"
        self.auto_btn.setText(f"⚡  Auto-Route:  {s}")

    def _clear_memory(self):
        self.memory.clear()
        self.mem_count.setText("  0 stored memories")
        self._insert_system_msg("Memory cleared.")

    def _new_chat(self):
        self.conversation.clear()
        self.chat_log.clear()
        self._welcome()

    def _save_transcript(self):
        # FIX-9: save to a predictable ~/jarvis_transcripts/ folder
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = pathlib.Path.home() / "jarvis_transcripts"
        dest.mkdir(exist_ok=True)
        path = dest / f"jarvis_transcript_{ts}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.conversation, fh, indent=2, ensure_ascii=False)
        self._insert_system_msg(f"Transcript saved → {path}")

    def _refresh_models(self):
        models = self.llm.get_available_models()
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
        else:
            self.model_combo.addItems(["qwen3:8b", "qwen3:14b"])

    # ── Stats (FIX-3: Ollama check runs in thread pool, never blocks UI) ───────

    def _poll_stats(self):
        """Kick off an async health check; update CPU/RAM immediately (no I/O)."""
        if _PSUTIL:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            self.cpu_lbl.setText(f"  CPU  : {cpu:.0f}%")
            self.ram_lbl.setText(f"  RAM  : {ram:.0f}%")

        task = _OllamaCheckTask(self.llm, self._check_signals)
        self._pool.start(task)

    def _apply_online_status(self, online: bool):
        """Called on the main thread when the background check finishes."""
        if online:
            self.status_lbl.setText("● Ollama online")
            self.status_lbl.setStyleSheet(f"color: {COL_GREEN}; font-family: Consolas; font-size: 8pt;")
            self.llm_lbl.setText("  LLM  : ONLINE")
        else:
            self.status_lbl.setText("○ Ollama offline")
            self.status_lbl.setStyleSheet("color: #883333; font-family: Consolas; font-size: 8pt;")
            self.llm_lbl.setText("  LLM  : OFFLINE")

    # ── System tray ────────────────────────────────────────────────────────────

    def _build_tray(self):
        pix = QPixmap(32, 32)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setPen(QColor(0, 212, 255))
        painter.setFont(QFont("Consolas", 15, QFont.Bold))
        painter.drawText(4, 24, "J")
        painter.end()

        self.tray = QSystemTrayIcon(QIcon(pix), self)
        menu = QMenu()
        menu.addAction("Show JARVIS", self.show)
        menu.addAction("Quit", self._quit)   # FIX-5: use clean _quit, not raw app.quit
        self.tray.setContextMenu(menu)
        self.tray.show()
        self.tray.activated.connect(
            lambda r: self.show() if r == QSystemTrayIcon.DoubleClick else None
        )

    # ── FIX-5: Clean shutdown ──────────────────────────────────────────────────

    def _quit(self):
        """Stop all timers, join the worker thread, then exit."""
        self._stats_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(2000)   # give it 2 s to finish gracefully
        QApplication.quit()

    def closeEvent(self, event):
        # Closing the window minimises to tray; quit via tray menu → _quit().
        event.ignore()
        self.hide()
        self.tray.showMessage(
            "JARVIS", "Running in system tray  —  double-click to restore.",
            QSystemTrayIcon.Information, 2000,
        )
