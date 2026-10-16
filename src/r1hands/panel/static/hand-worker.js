// Inference stays off the UI thread. The caller sends at most one frame at a time.
const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
let detector;
self.onmessage = async ({data}) => {
  if (data.type === "init") {
    try {
      importScripts(`${MP}/vision_bundle.js`);
      const {FilesetResolver, HandLandmarker} = vision;
      const files = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
      detector = await HandLandmarker.createFromOptions(files, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
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
