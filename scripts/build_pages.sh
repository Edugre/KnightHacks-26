#!/usr/bin/env bash
# Build the static site for Cloudflare Pages (build command: bash scripts/build_pages.sh, output directory: dist).
# Copies web/index.html into dist/ with its censor-api meta tag pointing at the server, so the page hosted at
# blurr.photo talks to the backend behind the named tunnel. The repo copy keeps the empty tag, which means "the
# server that served this page" (AUP's proxy, localhost). CENSOR_API overrides the URL.
set -euo pipefail
cd "$(dirname "$0")/.."
API="${CENSOR_API:-https://api.blurr.photo/}"
case "$API" in https://*|http://*) ;; *) echo "CENSOR_API must be a URL, got '$API'" >&2; exit 1 ;; esac
TAG='<meta name="censor-api" content="">'
grep -qF "$TAG" web/index.html || { echo "web/index.html has no empty censor-api meta tag to fill" >&2; exit 1; }
rm -rf dist && mkdir -p dist
sed "s|$TAG|<meta name=\"censor-api\" content=\"$API\">|" web/index.html > dist/index.html
echo "dist/index.html -> $API"
