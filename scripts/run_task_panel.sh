#!/usr/bin/env bash
# Local Mac simulation or a tunnel-accessible headless Linux server.
set -euo pipefail
mode="${1:-local}"
if [[ $# -gt 0 ]]; then shift; fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
case "$mode" in
  local) renderer="${R1H_TASK_RENDERER:-auto}" ;;
  server) renderer="${R1H_TASK_RENDERER:-egl}"; set -- --no-browser "$@" ;;
  *) echo 'Usage: scripts/run_task_panel.sh [local|server] [r1h-panel options]' >&2; exit 2 ;;
esac
if [[ ! -x .venv/bin/r1h-panel ]]; then
  echo "Install first: uv venv && uv pip install -e '.[panel,sim]'" >&2
  exit 1
fi
exec .venv/bin/r1h-panel --backend mock --sides right --port "${R1H_PANEL_PORT:-8081}" \
  --task-renderer "$renderer" "$@"
