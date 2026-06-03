"""
main.py — Jarvis entry point
Wires the animated boot screen to the main window.

FIX-10: Provides a proper entry point so LoadingScreen and JarvisWindow
        are created in the right order and on the main thread.

Usage:
    python main.py
"""
import sys

from PySide6.QtWidgets import QApplication

from loading_screen import LoadingScreen
from llm_router import LLMRouter
from memory import MemoryEngine
from main_ui import JarvisWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("JARVIS")
    app.setQuitOnLastWindowClosed(False)   # keep alive in tray after window closes

    # Initialise backend objects on the main thread before the worker threads
    # ever see them.  These constructors may print to stdout (progress info).
    llm = LLMRouter()
    mem = MemoryEngine()

    # Show the animated HUD boot screen first.
    splash = LoadingScreen()
    splash.show()

    # When the boot animation finishes, swap to the main window.
    def on_boot_finished():
        splash.close()
        win = JarvisWindow(llm, mem)
        # Keep a reference so Python's GC doesn't collect it.
        app._jarvis_win = win
        win.show()

    splash.finished.connect(on_boot_finished)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
