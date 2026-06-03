"""
loading_screen.py — Jarvis animated HUD boot screen
Sci-fi style: hex grid, scanning line, pulsing rings, side data panels.
Uses PySide6 QPainter — no external image assets needed.
"""
import math
import random
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush,
    QRadialGradient, QLinearGradient, QFontMetrics,
    QConicalGradient,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG         = QColor(2, 8, 12)
CYAN       = QColor(0, 212, 255)
CYAN_DIM   = QColor(0, 130, 160)
CYAN_FAINT = QColor(0, 212, 255, 30)
GREEN      = QColor(0, 255, 160)
WHITE      = QColor(210, 235, 248)
GRID_LINE  = QColor(0, 55, 70, 45)
DANGER     = QColor(255, 80, 80)

BOOT_PHASES = [
    ("CORE",      "Initializing neural processing core..."),
    ("LLM",       "Loading Qwen3 language model via Ollama..."),
    ("MEMORY",    "Connecting to ChromaDB vector database..."),
    ("FILE IDX",  "Scanning and embedding local file system..."),
    ("VOICE",     "Calibrating Whisper STT + Piper TTS engine..."),
    ("AGENT",     "Loading autonomous task-execution protocols..."),
    ("SECURITY",  "Establishing local-only sandbox environment..."),
    ("INTERFACE", "Rendering holographic user interface..."),
    ("DIAG",      "Running full system self-check..."),
    ("ONLINE",    "ALL SYSTEMS NOMINAL — JARVIS ONLINE"),
]


class LoadingScreen(QWidget):
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("JARVIS")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        screen = QApplication.primaryScreen().size()
        self.W, self.H = 920, 600
        self.setGeometry(
            (screen.width()  - self.W) // 2,
            (screen.height() - self.H) // 2,
            self.W, self.H,
        )

        # ── Animation state ───────────────────────────────────────
        self.scan_y       = 0
        self.ring_phase   = 0.0
        self.sweep        = 0.0
        self.progress     = 0
        self.phase_idx    = 0
        self.phase_ticks  = 0
        self.stats_ticks  = 0
        self.glitch       = 0        # glitch flash counter

        # Side panel data (updated ~1/s to avoid flicker)
        self.cpu   = 28
        self.ram   = 44
        self.temp  = 41
        self.nodes = 1024

        # Floating particles
        self.particles = [self._new_particle() for _ in range(55)]

        # Corner tick-marks (animated)
        self.corner_tick = 0

        # ── Fonts ─────────────────────────────────────────────────
        self.f_title = QFont("Consolas", 21, QFont.Bold)
        self.f_med   = QFont("Consolas", 10)
        self.f_small = QFont("Consolas", 9)
        self.f_tiny  = QFont("Consolas", 8)

        # ── Main timer (~33 fps) ──────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _new_particle(self):
        return {
            "x": random.uniform(0, self.W),
            "y": random.uniform(0, self.H),
            "vx": random.uniform(-0.22, 0.22),
            "vy": random.uniform(-0.22, 0.22),
            "a": random.randint(18, 75),
            "s": random.uniform(1.0, 2.4),
        }

    # ── Tick ───────────────────────────────────────────────────────────────────

    def _tick(self):
        W, H = self.W, self.H

        self.scan_y     = (self.scan_y + 3) % H
        self.ring_phase = (self.ring_phase + 0.042) % (2 * math.pi)
        self.sweep      = (self.sweep + 1.1) % 360
        self.corner_tick = (self.corner_tick + 1) % 120

        for p in self.particles:
            p["x"] = (p["x"] + p["vx"]) % W
            p["y"] = (p["y"] + p["vy"]) % H

        self.stats_ticks += 1
        if self.stats_ticks >= 33:
            self.stats_ticks = 0
            self.cpu   = max(12, min(88, self.cpu  + random.randint(-5, 5)))
            self.ram   = max(28, min(78, self.ram  + random.randint(-2, 2)))
            self.temp  = max(37, min(54, self.temp + random.randint(-1, 1)))
            self.nodes = max(512, min(4096, self.nodes + random.randint(-64, 64)))

        self.phase_ticks += 1
        if self.phase_ticks >= 33 and self.progress < 100:
            self.phase_ticks = 0
            self.progress = min(100, self.progress + 10)
            if self.phase_idx < len(BOOT_PHASES) - 1:
                self.phase_idx += 1
            if self.progress >= 100:
                self._timer.stop()
                QTimer.singleShot(1600, self.finished.emit)

        # Occasional glitch flash on last phase
        if self.progress == 100 and random.random() < 0.04:
            self.glitch = 3
        if self.glitch > 0:
            self.glitch -= 1

        self.update()

    # ── Draw helpers ───────────────────────────────────────────────────────────

    def _bg(self, p: QPainter):
        p.fillRect(self.rect(), BG)
        # subtle radial glow at centre
        cx, cy = self.W // 2, self.H // 2
        g = QRadialGradient(QPointF(cx, cy - 30), self.W * 0.55)
        g.setColorAt(0, QColor(0, 30, 45, 60))
        g.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), QBrush(g))

    def _hex_grid(self, p: QPainter):
        p.setPen(QPen(GRID_LINE, 0.6))
        r = 26
        dx = math.sqrt(3) * r
        dy = 1.5 * r
        cols = int(self.W / dx) + 3
        rows = int(self.H / dy) + 3
        for row in range(-1, rows):
            for col in range(-1, cols):
                cx = col * dx + (dx / 2 if row % 2 else 0)
                cy = row * dy
                pts = [
                    QPointF(cx + r * math.cos(math.radians(60 * i - 30)),
                             cy + r * math.sin(math.radians(60 * i - 30)))
                    for i in range(6)
                ]
                for i in range(6):
                    p.drawLine(pts[i], pts[(i + 1) % 6])

    def _scan_line(self, p: QPainter):
        y = self.scan_y
        W = self.W
        profile = [4, 8, 16, 35, 70, 110, 70, 35, 16, 8, 4]
        for i, a in enumerate(profile):
            col = QColor(0, 212, 255, a)
            p.setPen(QPen(col, 1))
            p.drawLine(0, y + i - 5, W, y + i - 5)
        # bright core
        p.setPen(QPen(QColor(180, 245, 255, 30), 1))
        p.drawLine(0, y, W, y)

    def _particles(self, p: QPainter):
        for pt in self.particles:
            col = QColor(0, 212, 255, int(pt["a"]))
            pen = QPen(col, pt["s"])
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawPoint(QPointF(pt["x"], pt["y"]))

    def _center_hud(self, p: QPainter):
        cx, cy = self.W // 2, self.H // 2 - 35

        # Three expanding/fading pulse rings
        for i in range(3):
            ph = (self.ring_phase + i * 1.05) % (2 * math.pi)
            r  = 36 + 58 * (ph / (2 * math.pi))
            a  = int(210 * (1.0 - ph / (2 * math.pi)))
            p.setPen(QPen(QColor(0, 212, 255, max(0, a)), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # Outer rotating dashed ring
        p.save()
        p.translate(cx, cy)
        p.rotate(self.sweep)
        pen = QPen(QColor(0, 160, 190, 130), 1)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(0, 0), 92, 92)
        p.restore()

        # Counter-rotating dot ring
        p.save()
        p.translate(cx, cy)
        p.rotate(-self.sweep * 0.65)
        pen2 = QPen(QColor(0, 255, 160, 80), 1)
        pen2.setStyle(Qt.DotLine)
        p.setPen(pen2)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(0, 0), 68, 68)
        p.restore()

        # Tick marks on outer ring (12 evenly spaced)
        p.save()
        p.translate(cx, cy)
        p.rotate(self.sweep * 0.3)
        for i in range(12):
            angle = math.radians(i * 30)
            x1 = 88 * math.cos(angle)
            y1 = 88 * math.sin(angle)
            x2 = (88 + (6 if i % 3 == 0 else 3)) * math.cos(angle)
            y2 = (88 + (6 if i % 3 == 0 else 3)) * math.sin(angle)
            p.setPen(QPen(CYAN if i % 3 == 0 else CYAN_DIM, 1.5 if i % 3 == 0 else 0.8))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        p.restore()

        # Glowing filled centre circle
        grad = QRadialGradient(QPointF(cx, cy), 40)
        grad.setColorAt(0,   QColor(0, 212, 255, 90))
        grad.setColorAt(0.5, QColor(0, 100, 150, 40))
        grad.setColorAt(1,   QColor(0,   0,   0,  0))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(CYAN, 2))
        p.drawEllipse(QPointF(cx, cy), 38, 38)

        # "J" glyph
        col = GREEN if self.glitch > 0 else WHITE
        p.setPen(QPen(col, 1))
        p.setFont(self.f_title)
        p.drawText(QRectF(cx - 24, cy - 22, 48, 44), Qt.AlignCenter, "J")

    def _corner_brackets(self, p: QPainter):
        W, H = self.W, self.H
        m, s = 16, 34
        # Animated inner tick
        tick_len = int(s * (self.corner_tick / 120))

        p.setPen(QPen(CYAN, 2))
        segs = [
            [(m + s, m), (m, m), (m, m + s)],
            [(W - m - s, m), (W - m, m), (W - m, m + s)],
            [(m + s, H - m), (m, H - m), (m, H - m - s)],
            [(W - m - s, H - m), (W - m, H - m), (W - m, H - m - s)],
        ]
        for seg in segs:
            for i in range(len(seg) - 1):
                x1, y1 = seg[i]
                x2, y2 = seg[i + 1]
                p.drawLine(x1, y1, x2, y2)

        # Animated corner tick (runs from corner outward)
        if tick_len > 0:
            p.setPen(QPen(GREEN, 1))
            p.drawLine(m, m, m + tick_len, m)
            p.drawLine(W - m, m, W - m - tick_len, m)
            p.drawLine(m, H - m, m + tick_len, H - m)
            p.drawLine(W - m, H - m, W - m - tick_len, H - m)

    def _side_panels(self, p: QPainter):
        W, H = self.W, self.H

        left = [
            ("◆ SYSTEM DIAGNOSTICS", GREEN),
            (f"  CPU LOAD  : {self.cpu:>3}%",  CYAN_DIM),
            (f"  RAM USAGE : {self.ram:>3}%",  CYAN_DIM),
            (f"  CPU TEMP  : {self.temp:>3}°C", CYAN_DIM),
            ("  GPU ACCEL : OFFLINE",           QColor(80, 80, 80)),
            ("  DISK I/O  : NOMINAL",           CYAN_DIM),
        ]
        right = [
            ("◆ JARVIS MODULES", GREEN),
            ("  LLM ENGINE : QWEN3-8B", CYAN_DIM),
            ("  MEMORY DB  : CHROMADB", CYAN_DIM),
            ("  VOICE STT  : WHISPER",  CYAN_DIM),
            ("  VOICE TTS  : PIPER",    CYAN_DIM),
            (f"  NODES      : {self.nodes}", CYAN_DIM),
        ]

        fm = QFontMetrics(self.f_tiny)
        p.setFont(self.f_tiny)

        y_start = 158
        for i, (text, color) in enumerate(left):
            p.setPen(QPen(color, 1))
            p.drawText(28, y_start + i * 20, text)

        for i, (text, color) in enumerate(right):
            p.setPen(QPen(color, 1))
            tw = fm.horizontalAdvance(text)
            p.drawText(W - 28 - tw, y_start + i * 20, text)

    def _title_block(self, p: QPainter):
        W, H = self.W, self.H
        mid_y = H // 2 + 55

        col = GREEN if self.glitch > 0 else CYAN
        p.setPen(QPen(col, 1))
        p.setFont(self.f_title)
        p.drawText(QRectF(0, mid_y, W, 38), Qt.AlignCenter, "J.A.R.V.I.S")

        p.setPen(QPen(QColor(0, 100, 130), 1))
        p.setFont(self.f_tiny)
        p.drawText(QRectF(0, mid_y + 42, W, 18), Qt.AlignCenter,
                   "JUST A RATHER VERY INTELLIGENT SYSTEM  ·  LOCAL OFFLINE AI  ·  v1.0")

    def _boot_phase(self, p: QPainter):
        W, H = self.W, self.H
        if self.phase_idx >= len(BOOT_PHASES):
            return
        tag, msg = BOOT_PHASES[self.phase_idx]

        p.setPen(QPen(GREEN, 1))
        p.setFont(QFont("Consolas", 9, QFont.Bold))
        p.drawText(QRectF(0, H - 118, W, 20), Qt.AlignCenter, f"[ {tag} ]")

        p.setPen(QPen(CYAN_DIM, 1))
        p.setFont(self.f_tiny)
        p.drawText(QRectF(0, H - 97, W, 18), Qt.AlignCenter, msg)

    def _progress_bar(self, p: QPainter):
        W, H = self.W, self.H
        bw = int(W * 0.62)
        bh = 3
        bx = (W - bw) // 2
        by = H - 68

        # Track
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 30, 40)))
        p.drawRect(bx, by, bw, bh)

        # Filled portion
        fw = int(bw * self.progress / 100)
        if fw > 0:
            g = QLinearGradient(bx, 0, bx + bw, 0)
            g.setColorAt(0, QColor(0, 110, 150))
            g.setColorAt(1, CYAN)
            p.setBrush(QBrush(g))
            p.drawRect(bx, by, fw, bh)

            # Glowing tip
            tip = float(bx + fw)
            glow = QRadialGradient(QPointF(tip, by + 1.5), 11)
            glow.setColorAt(0, QColor(0, 212, 255, 200))
            glow.setColorAt(1, QColor(0,   0,   0,   0))
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(tip, by + 1.5), 11, 11)

        # Labels
        p.setPen(QPen(CYAN, 1))
        p.setFont(self.f_small)
        p.drawText(bx + bw + 10, by + 10, f"{self.progress}%")

        p.setPen(QPen(QColor(0, 70, 90), 1))
        p.setFont(self.f_tiny)
        p.drawText(bx, by + 18,
                   f"BOOT SEQUENCE  {self.phase_idx + 1} / {len(BOOT_PHASES)}")

    # ── Paint ──────────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            self._bg(p)
            self._hex_grid(p)
            self._particles(p)
            self._scan_line(p)
            self._corner_brackets(p)
            self._center_hud(p)
            self._side_panels(p)
            self._title_block(p)
            self._boot_phase(p)
            self._progress_bar(p)
        finally:
            p.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._timer.stop()
            self.finished.emit()


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    screen = LoadingScreen()
    screen.finished.connect(app.quit)
    screen.show()
    sys.exit(app.exec())
