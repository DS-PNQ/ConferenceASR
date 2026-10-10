"""Render electron/build/icon.svg -> icon.png + icon.ico (needs rsvg-convert on PATH, and Pillow)."""
import io
import subprocess
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "electron" / "build"

subprocess.run(["rsvg-convert", "-w", "512", "-h", "512", OUT / "icon.svg", "-o", OUT / "icon.png"], check=True)
big = subprocess.run(["rsvg-convert", "-w", "256", "-h", "256", OUT / "icon.svg"], check=True, capture_output=True).stdout
Image.open(io.BytesIO(big)).save(OUT / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
print("wrote", OUT / "icon.png", OUT / "icon.ico")
