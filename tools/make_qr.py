"""Generate the QR codes that open the AR menu page.

  qr.png  large, print/screen quality - scan this with a phone camera
  qr.svg  embedded on the page for desktop visitors

Run:  .venv/Scripts/python tools/make_qr.py [url]
"""
import sys
from pathlib import Path

import segno

ROOT = Path(__file__).resolve().parent.parent
URL = sys.argv[1] if len(sys.argv) > 1 else "https://pmtchakanyuka-star.github.io/ar-menu-poc/"

qr = segno.make(URL, error="h")  # high error correction: survives smudges, glare, a logo overlay
qr.save(ROOT / "qr.png", scale=24, border=4, dark="#000000", light="#ffffff")
qr.save(ROOT / "qr.svg", scale=1, border=2, dark="#2b1a12", light="#ffffff")
print(f"{URL}  ->  QR version {qr.designator}")
