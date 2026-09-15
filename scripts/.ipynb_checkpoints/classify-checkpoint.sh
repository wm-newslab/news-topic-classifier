#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$repo_dir/src/classification/classify_uslnda_news_parallel.py" --adapter-dir "$repo_dir/models/adapter" "$@"
