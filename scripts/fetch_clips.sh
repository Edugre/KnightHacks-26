#!/usr/bin/env bash
# Download the Pexels benchmark clips listed in bench/clips/manifest.yaml (Pexels license, free to use).
# Videos are gitignored, so run this once per clone that needs them. Needs network and ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/../bench/clips"
FFMPEG="${CENSOR_FFMPEG:-ffmpeg}"
get() {  # get <pexels id> <file name on videos.pexels.com> <output>
  [ -s "$3" ] && { echo "have $3"; return; }
  curl -sSL -A "Mozilla/5.0" -o "$3" "https://videos.pexels.com/video-files/$1/$2"
  echo "got $3"
}
get 4625295 4625295-hd_1920_1080_25fps.mp4 street_talk.mp4
get 4625331 4625331-hd_1920_1080_25fps.mp4 street_walk.mp4
if [ ! -s street_plates.mp4 ]; then
  get 5845229 5845229-hd_1920_1080_25fps.mp4 street_plates_full.mp4
  "$FFMPEG" -y -loglevel error -i street_plates_full.mp4 -t 13 -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -an street_plates.mp4
  rm street_plates_full.mp4
  echo "trimmed street_plates.mp4 to 13 s"
fi
