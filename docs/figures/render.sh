#!/usr/bin/env bash
# Render each page in docs/figures/src to a PDF with headless Chrome, which shapes Devanagari,
# Nastaliq, Tamil and Ge'ez text correctly, crop it to its drawn content, and rasterize a 2x PNG.
# Run `python -m lost_in_injection.tables` first so the table pages hold the latest numbers.
# Usage: docs/figures/render.sh [page ...]   (default: every page in src/)
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
pages=("$@")
[ ${#pages[@]} -gt 0 ] || pages=($(ls src/*.html | xargs -n1 basename | sed 's/\.html$//'))
for page in "${pages[@]}"; do
  "$CHROME" --headless=new --disable-gpu --allow-file-access-from-files --virtual-time-budget=3000 \
    --no-pdf-header-footer --print-to-pdf="$PWD/$page.pdf" "file://$PWD/src/$page.html" 2> /dev/null
  uv run --no-project -q --with pymupdf --with pillow python -c '
import sys, fitz
from PIL import Image, ImageOps
path, png = sys.argv[1], sys.argv[2]
pdf = fitz.open(path)
assert len(pdf) == 1, f"{path} spills onto {len(pdf)} pages; raise its @page size"
page = pdf[0]
low = page.get_pixmap(dpi=96)  # One pixel per CSS px; find the drawn content from the pixels
image = Image.frombytes("RGB", (low.width, low.height), low.samples).convert("L")
left, top, right, bottom = ImageOps.invert(image).getbbox()
box = (fitz.Rect(left, top, right, bottom) * (72 / 96) + (-3, -3, 3, 3)) & page.rect
page.set_cropbox(box)
page.get_pixmap(dpi=192).save(png)  # The cropped page; Chrome prints 96 px per inch, so 2x
pdf.save(path + ".tmp")
' "$page.pdf" "$page.png"
  mv "$page.pdf.tmp" "$page.pdf"
  echo "rendered $page"
done
