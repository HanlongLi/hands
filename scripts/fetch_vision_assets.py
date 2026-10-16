#!/usr/bin/env python3
"""Reproduce the pinned, locally served MediaPipe assets (network needed only here)."""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import tarfile
from urllib.request import urlopen

VERSION = '0.10.14'
PACKAGE_URL = f'https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{VERSION}.tgz'
PACKAGE_SHA512 = 'vOifgZhkndgybdvoRITzRkIueWWSiCKuEUXXK6Q4FaJsFvRJuwgg++vqFUMlL0Uox62U5aEXFhHxlhV7Ja5e3Q=='
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
DESTINATION = Path(__file__).resolve().parents[1] / 'src/r1hands/panel/static/vendor/mediapipe'
FILES = ('vision_bundle.cjs', 'wasm/vision_wasm_internal.js',
         'wasm/vision_wasm_internal.wasm', 'wasm/vision_wasm_nosimd_internal.js',
         'wasm/vision_wasm_nosimd_internal.wasm')


def download(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    archive = download(PACKAGE_URL)
    if base64.b64encode(hashlib.sha512(archive).digest()).decode() != PACKAGE_SHA512:
        raise ValueError('MediaPipe npm archive integrity mismatch')
    model = download(MODEL_URL)
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise ValueError('Hand model integrity mismatch')
    # Select files explicitly; never extract arbitrary archive paths.
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as package:
        for name in FILES:
            path = DESTINATION / ("vision_bundle.js" if name == "vision_bundle.cjs" else name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(package.extractfile(f'package/{name}').read())
    (DESTINATION / 'hand_landmarker.task').write_bytes(model)
    print(f'Installed MediaPipe {VERSION} and hand model in {DESTINATION}')


if __name__ == '__main__':
    main()
