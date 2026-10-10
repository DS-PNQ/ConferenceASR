"""Render the ConfLive icon -> electron/build/icon.png + icon.ico (needs Pillow)."""
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
OUT = Path(__file__).resolve().parent.parent / "electron" / "build"

grad = Image.new("RGB", (S, S))
g = ImageDraw.Draw(grad)
top, bot = (0x47, 0x47, 0x47), (0x14, 0x14, 0x14)  # same charcoal as the app's pill buttons
for y in range(S):
    t = y / S
    g.line([(0, y), (S, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bot)))
mask = Image.new("L", (S, S))
ImageDraw.Draw(mask).rounded_rectangle([40, 40, S - 40, S - 40], radius=230, fill=255)
img = Image.new("RGBA", (S, S))
img.paste(grad, mask=mask)

d = ImageDraw.Draw(img)
for i, h in enumerate((220, 400, 580, 400, 220)):                            # live waveform
    x = 272 + i * 100
    d.rounded_rectangle([x, 512 - h // 2, x + 72, 512 + h // 2], radius=36, fill="white")

OUT.mkdir(parents=True, exist_ok=True)
img.resize((512, 512), Image.LANCZOS).save(OUT / "icon.png")
img.save(OUT / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
print("wrote", OUT / "icon.png", OUT / "icon.ico")
