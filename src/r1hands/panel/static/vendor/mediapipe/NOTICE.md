# MediaPipe runtime assets

Vendored from Google's `@mediapipe/tasks-vision` npm package, version **0.10.14**.
Package code and WASM are covered by the accompanying Apache-2.0 LICENSE.

- Package: https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-0.10.14.tgz
- Source: https://github.com/google-ai-edge/mediapipe/tree/v0.10.14
- Hand model: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- Model SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`

Run `python scripts/fetch_vision_assets.py` to reproduce the selected runtime files.
The CommonJS bundle is served as `vision_bundle.js` so Python serves a JavaScript MIME
type; its contents are unchanged. The script verifies the npm package's pinned SHA-512 integrity and the model SHA-256
before writing files. No CDN or model download occurs in the browser at runtime.
