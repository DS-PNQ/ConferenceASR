"""Launch the ConfLive desktop app (native window, no browser).

Usage:
    python desktop_app.py
"""
import logging

from desktop.bootstrap import build_app
from desktop.ui import ConferenceApp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

if __name__ == "__main__":
    ctx = build_app()
    app = ConferenceApp(ctx)
    app.mainloop()
