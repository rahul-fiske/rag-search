"""Synthetic scanned pages with known text and controlled damage (Pillow; imported lazily by callers).

The routing harness needs pages whose true text is known exactly and whose quality can be turned down one notch at a
time: resolution, blur, skew, speckle, JPEG compression, a ruled table, a photograph-like background.  Every page is
made from a list of text lines drawn with Pillow's scalable default font, so no personal document is involved.  A page
is an image; ``write_pdf`` wraps images into an image-only PDF (no text layer), the thing a scanner produces.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

PAGE_PT = (595, 842)                 # A4 in points
WORDS = ("storage array volume snapshot replication cluster node port fabric zone host adapter firmware policy "
         "quota capacity latency throughput pool mirror stripe parity controller enclosure drive bay power cooling "
         "invoice payment account balance statement deposit withdrawal interest customer agreement clause party "
         "schedule notice amount tax return filing assessment certificate").split()


def text_lines(seed: int, n: int = 38, words: int = 11) -> list[str]:
    """*n* plausible lines of English-looking text, repeatable from *seed*."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ws = [rng.choice(WORDS) for _ in range(words)]
        ws[0] = ws[0].capitalize()
        out.append(" ".join(ws) + ("." if rng.random() < 0.5 else ""))
    return out


def make_page(lines: list[str], *, dpi: int = 200, blur: float = 0.0, skew: float = 0.0, noise: float = 0.0,
              jpeg: int = 0, table: bool = False, photo: bool = False, seed: int = 0) -> Any:
    """A greyscale page image of *lines* at *dpi*, damaged as asked: Gaussian *blur* radius (pixels), *skew* in
    degrees, *noise* as the share of speckled pixels, *jpeg* quality (0 = none), a ruled *table* grid over the lower
    half, a *photo*-like textured background."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = round(PAGE_PT[0] * dpi / 72), round(PAGE_PT[1] * dpi / 72)
    rng = random.Random(seed)
    img = Image.new("L", (w, h), 255)
    if photo:                                          # a smooth, noisy, mid-tone texture: not a clean document
        noise_img = Image.effect_noise((w, h), 60).convert("L")
        img = Image.blend(Image.new("L", (w, h), 140), noise_img, 0.9)
    d = ImageDraw.Draw(img)
    size = max(8, round(11 * dpi / 72))
    font = ImageFont.load_default(size=size)
    margin = round(0.9 * dpi)
    y = margin
    step = round(size * 1.45)
    for line in lines:
        if y > h - margin:
            break
        d.text((margin, y), line, fill=0, font=font)
        y += step
    if table:
        top = h // 2
        for i in range(8):
            yy = top + i * step * 2
            d.line((margin, yy, w - margin, yy), fill=0, width=max(1, dpi // 100))
        for i in range(5):
            xx = margin + i * (w - 2 * margin) // 4
            d.line((xx, top, xx, top + 14 * step), fill=0, width=max(1, dpi // 100))
    if skew:
        img = img.rotate(skew, resample=Image.BICUBIC, expand=False, fillcolor=255)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        px = img.load()
        for _ in range(int(noise * w * h)):
            px[rng.randrange(w), rng.randrange(h)] = rng.choice((0, 255))
    if jpeg:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=jpeg)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img


def write_pdf(path: Path, images: list[Any], dpi: int = 200) -> None:
    """An image-only PDF: one page per image, no text layer."""
    images[0].save(path, "PDF", resolution=float(dpi), save_all=True, append_images=images[1:])


DAMAGE = {                       # name -> make_page keywords: a ladder from a clean scan to hard cases
    "clean": {},
    "dpi150": {"dpi": 150},
    "dpi100": {"dpi": 100},
    "blur1": {"blur": 1.2},
    "blur2": {"blur": 2.5},
    "skew1": {"skew": 1.0},
    "skew4": {"skew": 4.0},
    "noise": {"noise": 0.02},
    "jpeg20": {"jpeg": 20},
    "table": {"table": True},
    "photo": {"photo": True},
}
