"""Render the ConfLive icon -> electron/build/icon.png + icon.ico (needs Pillow)."""
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
OUT = Path(__file__).resolve().parent.parent / "electron" / "build"

grad = Image.new("RGB", (S, S))
g = ImageDraw.Draw(grad)
top, bot = (0x7C, 0x78, 0xE0), (0x1D, 0x26, 0x40)  # app purple -> sidebar navy
for y in range(S):
    t = y / S
    g.line([(0, y), (S, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bot)))
mask = Image.new("L", (S, S))
ImageDraw.Draw(mask).rounded_rectangle([40, 40, S - 40, S - 40], radius=230, fill=255)
img = Image.new("RGBA", (S, S))
img.paste(grad, mask=mask)

d = ImageDraw.Draw(img)
d.rounded_rectangle([200, 240, 824, 700], radius=150, fill="white")          # speech bubble
d.polygon([(330, 660), (300, 830), (480, 690)], fill="white")                # its tail
for i, h in enumerate((90, 200, 300, 200, 90)):                              # live waveform
    x = 335 + i * 90
    d.rounded_rectangle([x, 470 - h // 2, x + 54, 470 + h // 2], radius=27, fill=(0x63, 0x5F, 0xBD))

OUT.mkdir(parents=True, exist_ok=True)
img.resize((512, 512), Image.LANCZOS).save(OUT / "icon.png")
img.save(OUT / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
print("wrote", OUT / "icon.png", OUT / "icon.ico")
