"""Regression for the missing CDN bundle: all runtime assets must ship locally."""
import hashlib
from pathlib import Path

from r1hands.panel.app import STATIC


def test_vision_runtime_is_complete_and_model_matches_pinned_hash():
    root = STATIC / 'vendor' / 'mediapipe'
    bundle = (root / 'vision_bundle.js').read_text()
    assert 'exports.HandLandmarker=' in bundle
    assert 'exports.FilesetResolver=' in bundle
    for variant in ('internal', 'nosimd_internal'):
        assert (root / 'wasm' / f'vision_wasm_{variant}.js').stat().st_size > 1000
        assert (root / 'wasm' / f'vision_wasm_{variant}.wasm').read_bytes()[:4] == b'\x00asm'
    model = (root / 'hand_landmarker.task').read_bytes()
    assert hashlib.sha256(model).hexdigest() == 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
    assert (root / 'LICENSE').is_file()


def test_worker_loads_only_local_runtime_assets():
    source = (STATIC / 'hand-worker.js').read_text()
    assert 'https://' not in source
    assert 'self.exports = {}' in source
    assert 'const {FilesetResolver, HandLandmarker} = self.exports' in source
