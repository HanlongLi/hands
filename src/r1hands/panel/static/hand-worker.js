// Inference stays off the UI thread. The caller sends at most one frame at a time.
const MP = "/static/vendor/mediapipe";
let detector;
self.onmessage = async ({data}) => {
  if (data.type === "init") {
    try {
      // The pinned npm package ships CommonJS and ESM, not vision_bundle.js.
      // A classic worker is required by its WASM loader's importScripts path.
      self.exports = {};
      importScripts(`${MP}/vision_bundle.js`);
      const {FilesetResolver, HandLandmarker} = self.exports;
      const files = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
      detector = await HandLandmarker.createFromOptions(files, {
        baseOptions: {
          modelAssetPath: `${MP}/hand_landmarker.task`,
          delegate: "CPU",
        },
        runningMode: "VIDEO", numHands: 2,
        minHandDetectionConfidence: 0.7, minHandPresenceConfidence: 0.7,
        minTrackingConfidence: 0.7,
      });
      self.postMessage({type: "ready"});
    } catch (error) { self.postMessage({type: "error", error: String(error)}); }
    return;
  }
  if (data.type === "frame") {
    const started = performance.now();
    try {
      const result = detector.detectForVideo(data.bitmap, data.timestamp);
      self.postMessage({type: "result", timestamp: data.timestamp,
        inference_ms: performance.now() - started,
        hands: result.worldLandmarks.map((points, i) => ({
          points: points.map(p => [p.x, p.y, p.z]),
          image: result.landmarks[i],
          label: result.handednesses[i][0].categoryName.toLowerCase(),
          score: result.handednesses[i][0].score,
        }))});
    } catch (error) { self.postMessage({type: "error", error: String(error)}); }
    finally { data.bitmap.close(); }
  }
};
