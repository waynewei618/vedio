#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./download_bilibili_hd.sh BV1BbUSB4EGN

Environment:
  YTDLP    Override yt-dlp executable path.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

video_id="$1"

if [[ ! "$video_id" =~ ^(BV[0-9A-Za-z]+|av[0-9]+)$ ]]; then
  echo "error: expected a Bilibili video id, e.g. BV1BbUSB4EGN" >&2
  exit 2
fi

ytdlp="${YTDLP:-/home/sil/workspace/conda_envs/veddo/bin/yt-dlp}"

if [[ ! -x "$ytdlp" ]]; then
  echo "error: yt-dlp not found or not executable: $ytdlp" >&2
  echo "hint: install yt-dlp in the conda veddo environment, or set YTDLP=/path/to/yt-dlp" >&2
  exit 1
fi

url="https://www.bilibili.com/video/${video_id}/"

"$ytdlp" \
  --cookies-from-browser chrome \
  -f "bv*+ba/b" \
  --merge-output-format mp4 \
  -o "%(title).180B [%(id)s] [%(height)sp].%(ext)s" \
  "$url"
