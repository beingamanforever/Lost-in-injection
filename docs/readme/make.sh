#!/usr/bin/env bash
# README assets: the logo (SVG with text as paths, and a 2x PNG, one image for light and dark), rendered by
# headless Chrome on a transparent page, and short GIF loops cut from the rendered demo video.
# Usage: docs/readme/make.sh [path/to/LostInInjection.mp4]
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --disable-gpu --virtual-time-budget=5000 --no-pdf-header-footer \
  --print-to-pdf="$PWD/logo.pdf" "file://$PWD/src/logo.html" 2> /dev/null
uv run --no-project -q --with pymupdf python -c '
import fitz
page = fitz.open("logo.pdf")[0]
low = page.get_pixmap(dpi=96, alpha=True)  # One pixel per CSS px; crop to what is drawn
alpha = low.samples[3::4]
rows = [y for y in range(low.height) if any(alpha[y * low.width:(y + 1) * low.width])]
cols = [x for x in range(low.width) if any(alpha[x::low.width])]
page.set_cropbox((fitz.Rect(cols[0], rows[0], cols[-1] + 1, rows[-1] + 1) * (72 / 96) + (-2, -2, 2, 2)) & page.rect)
page.get_pixmap(dpi=192, alpha=True).save("logo.png")
open("logo.svg", "w").write(page.get_svg_image(text_as_path=True))
'
rm logo.pdf
video="${1:-}"
[ -n "$video" ] || exit 0
FFMPEG="$(uv run --no-project -q --with imageio-ffmpeg python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
gif() {  # name start duration [fps]: 800 px wide, one palette per clip
  "$FFMPEG" -loglevel error -y -ss "$2" -t "$3" -i "$video" -filter_complex \
    "fps=${4:-20},scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none" "$1.gif"
}
gif attack 0.6 7.4
gif transfer 37.2 5.6
gif best-of-all 43.45 7.3 12  # Thousands of cells change colour, so fewer frames
