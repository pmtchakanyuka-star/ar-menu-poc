"""Detect calibration edges in the reference photo by casting rays from the cake.

Writes tools/calib/edges.json with image-space points for:
  plate_rim   - outer edge of the white plate (white -> green carrier)
  pool_edge   - edge of the glaze pooled on the plate (dark -> white plate)
  top_outline - far silhouette of the cake top against the countertop
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
PHOTO = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "reference.jpg"

img = Image.open(PHOTO).convert("RGB")
W, H = img.size
px = np.asarray(img.filter(ImageFilter.BoxBlur(3))).astype(float)
px5 = np.asarray(img.filter(ImageFilter.BoxBlur(6))).astype(float)


def classify(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = a.max(-1) - a.min(-1)
    green = (g > r + 25) & (g > b + 60)
    white = (lum > 195) & (sat < 50) & ~green
    dark = lum < 110
    return green, white, dark, lum


def ray(x0, y0, ang, n=700, step=1.0):
    t = np.arange(n) * step
    xs, ys = x0 + t * np.cos(ang), y0 + t * np.sin(ang)
    ok = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
    return xs[ok], ys[ok]


def first_run(mask, start, length):
    """Index of the first run of `length` True values at or after `start`."""
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= length:
            return i - length + 1
    return None


plate_rim, pool_edge, top_outline = [], [], []
cx, cy = 660.0, 1050.0
for deg in np.arange(-150, 211, 2.0):
    xs, ys = ray(cx, cy, np.radians(deg))
    green, white, dark, _ = classify(px[ys.astype(int), xs.astype(int)])
    # Find the green carrier first, then walk back across the white plate band:
    # glossy highlights on the glaze also look "white", but they are not
    # followed by the green carrier.
    i_green = first_run(green, 0, 4)
    if i_green is None:
        continue
    j = i_green - 1
    while j > 0 and not white[j] and i_green - j < 12:      # small blur gap
        j -= 1
    if not white[j]:
        continue
    while j > 0 and (white[j] or white[max(0, j - 1)]):       # back across the band
        j -= 1
    band = i_green - j
    if not (6 <= band <= 140):
        continue
    plate_rim.append((float(xs[i_green]), float(ys[i_green]), float(deg)))
    if dark[max(0, j - 30):j + 1].any():                    # band starts at the glaze
        pool_edge.append((float(xs[j + 1]), float(ys[j + 1]), float(deg)))

hx, hy = 677.0, 680.0  # hole centre (approx.)
for deg in np.arange(-178, -1, 2.0):
    xs, ys = ray(hx, hy, np.radians(deg), n=900)
    _, white, dark, lum = classify(px5[ys.astype(int), xs.astype(int)])
    dist = np.hypot(xs - hx, ys - hy)
    light = lum > 135
    start = int(np.searchsorted(dist, 140))  # skip the hole and its far wall
    i = first_run(light, start, 7)
    if i is not None and dark[max(0, i - 20):i].mean() > 0.4:
        top_outline.append((float(xs[i]), float(ys[i]), float(deg)))

out = {"image_size": [W, H], "plate_rim": plate_rim, "pool_edge": pool_edge, "top_outline": top_outline}
(HERE / "edges.json").write_text(json.dumps(out, indent=1))
print(f"plate_rim {len(plate_rim)}  pool_edge {len(pool_edge)}  top_outline {len(top_outline)}")

ov = img.copy()
d = ImageDraw.Draw(ov)
for pts, col in ((plate_rim, (0, 140, 255)), (pool_edge, (255, 0, 200)), (top_outline, (255, 230, 0))):
    for x, y, _ in pts:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=col, width=2)
ov.resize((600, 800)).save(HERE / "edges_overlay.png")
