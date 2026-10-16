#!/usr/bin/env bash
# Push this repo from the Mac to the R1 development computer and set up the level-1 (hands) runtime there.
#   scripts/deploy_jetson.sh [ssh-host]        default: r1  (~/.ssh/config: unitree@192.168.123.164)
#
# The R1 computer has no internet and only Python 3.8 (Ubuntu 20.04, aarch64, glibc 2.31). So the Mac
# downloads a standalone CPython 3.11 (python-build-standalone) and aarch64 wheels once, caches them in
# .deploy_cache/, and ships them. On the robot: ~/r1hands_runtime/python (interpreter), ~/r1hands/.venv
# (venv, offline pip install), and a .pth file that puts ~/r1hands/src on sys.path. The system Python is
# never touched. Re-running only re-syncs the code.
set -euo pipefail
TARGET="${1:-r1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.deploy_cache"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-aarch64-unknown-linux-gnu-install_only.tar.gz"
PLAT=(--platform manylinux_2_31_aarch64 --platform manylinux_2_28_aarch64 --platform manylinux_2_17_aarch64 --platform manylinux2014_aarch64 --platform any)

mkdir -p "$CACHE/wheels"
[ -f "$CACHE/py311.tar.gz" ] || curl -sfL -o "$CACHE/py311.tar.gz" "$PY_URL"
python3 -m pip download -q --dest "$CACHE/wheels" --only-binary=:all: --python-version 3.11 --implementation cp "${PLAT[@]}" \
  "bc-stark-sdk>=2.0.7" "numpy>=1.24,<2.3" "pyyaml>=6" "aiohttp>=3.9"

rsync -az --delete --exclude .venv --exclude .git --exclude .deploy_cache --exclude __pycache__ --exclude logs \
  --exclude '*.profile' --exclude 'assets/r1_revo2' --exclude 'assets/*description/meshes' "$ROOT/" "$TARGET:r1hands/"
ssh "$TARGET" 'mkdir -p ~/r1hands_runtime/wheels'
rsync -az "$CACHE/py311.tar.gz" "$TARGET:r1hands_runtime/"
rsync -az "$CACHE/wheels/" "$TARGET:r1hands_runtime/wheels/"

ssh "$TARGET" 'set -e
  RT=~/r1hands_runtime
  [ -x $RT/python/bin/python3.11 ] || tar -xzf $RT/py311.tar.gz -C $RT
  cd ~/r1hands
  [ -x .venv/bin/python ] || $RT/python/bin/python3.11 -m venv .venv
  .venv/bin/pip install -q --no-index --find-links $RT/wheels "bc-stark-sdk>=2.0.7" numpy pyyaml aiohttp
  SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
  echo "$HOME/r1hands/src" > "$SITE/r1hands_src.pth"
  .venv/bin/python -c "from bc_stark_sdk import main_mod; import r1hands, aiohttp, numpy, yaml; print(\"runtime OK:\", r1hands.__file__)"
  groups | grep -qw dialout || echo "NOTE: serial access needs: sudo usermod -aG dialout $USER  (then log out/in)"
  ls /dev/ttyUSB* /dev/ttyCH343USB* /dev/revo2_* 2>/dev/null || echo "no hand serial devices visible yet"'
echo "deployed to $TARGET:~/r1hands"
