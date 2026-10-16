(function () {
  let hooks, recorder = null, clip = null, playback = null, stream = 0;
  const pageId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  let lastStats = [];
  const $ = id => document.getElementById(id);
  function leaveReview() {
    $('preview-mode').value = 'mapping';
    window.dispatchEvent(new CustomEvent('r1hands-review-reset'));
  }
  function message(text) { $('clip-status').textContent = text; }
  function stopRecording() {
    if (!recorder) return;
    clip = recorder.clip; recorder = null;
    $('landmark-record').textContent = 'Record landmarks';
    if (clip.frames.length) {
      playback = new LandmarkClip.Playback(clip);
      message(`${clip.frames.length} frames · ${(clip.frames.at(-1).t / 1000).toFixed(1)} s · ready to replay or download`);
    } else { clip = null; message('No frames captured. Start the camera and try again.'); }
  }
  function pause() {
    const wasPlaying = !!playback?.playing;
    if (playback) playback.pause(performance.now());
    if (wasPlaying) message('Replay paused · preview held');
    $('clip-play').textContent = 'Replay in preview';
  }
  function install(value) {
    const validated = LandmarkClip.validate(value);
    stopRecording(); pause(); leaveReview();
    clip = validated; playback = new LandmarkClip.Playback(clip);
    message(`${clip.synthetic ? 'Synthetic example · ' : ''}${clip.frames.length} frames · ${(clip.frames.at(-1).t / 1000).toFixed(1)} s`);
    $('clip-progress').max = clip.frames.length - 1; $('clip-progress').value = 0;
  }
  function download() {
    if (!clip?.frames.length) { message('Record or load a clip first.'); return; }
    const blob = new Blob([JSON.stringify(clip)], {type: 'application/json'});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = `right-hand-landmarks-${Date.now()}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  window.DemoLab = {
    attach(value) {
      hooks = value;
      $('landmark-record').onclick = () => {
        if (recorder) { stopRecording(); return; }
        if (!hooks.cameraOn()) { message('Start the camera before recording landmarks.'); return; }
        pause(); leaveReview(); recorder = new LandmarkClip.Recorder(hooks.state().retarget_profile, $('swap').checked);
        recorder.clip.metadata = {operator_id: $('mapping-operator').value.trim(), session_id: $('mapping-session').value.trim()};
        $('landmark-record').textContent = 'Stop landmark recording'; message('Recording raw landmarks, including tracking gaps · maximum 2 min');
      };
      $('clip-download').onclick = () => { stopRecording(); download(); };
      $('clip-load').onchange = async event => {
        const file = event.target.files[0];
        if (!file) return;
        try {
          if (file.size > 20 * 1024 * 1024) throw new Error('Clip exceeds 20 MiB');
          install(JSON.parse(await file.text()));
        } catch (error) { message(`Could not load clip: ${error.message}`); }
        event.target.value = '';
      };
      $('clip-example').onclick = async () => {
        try {
          const response = await fetch('/preview/example');
          if (!response.ok) throw new Error('Example clip unavailable');
          install(await response.json());
        } catch (error) { message(error.message); }
      };
      $('clip-play').onclick = () => {
        stopRecording();
        if (!playback) { message('Record, import, or load an example first.'); return; }
        if (playback.playing) { pause(); return; }
        if (!hooks.connected()) { message('Wait for the panel connection before replaying.'); return; }
        hooks.stopCamera(); leaveReview();
        if (playback.index === playback.clip.frames.length) playback.rewind();
        if (playback.index === 0) stream++;
        playback.play(performance.now());
        $('clip-play').textContent = 'Pause replay';
        message('Replaying landmarks into the 3D preview only · no motor commands');
      };
      $('clip-rewind').onclick = () => { pause(); leaveReview(); if (playback) {playback.rewind(); stream++;} $('clip-progress').value = 0; };
      $('clip-profile').onchange = () => { pause(); leaveReview(); if (playback) {playback.rewind(); stream++;} };
      document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
      window.addEventListener('r1hands-state', ({detail}) => {
        const target = detail.preview?.target;
        if (target) {
          lastStats.push({t: performance.now(), q: target});
          lastStats = lastStats.filter(row => performance.now() - row.t < 2000);
          const std = target.map((_, i) => {
            const mean = lastStats.reduce((n, row) => n + row.q[i], 0) / lastStats.length;
            return Math.sqrt(lastStats.reduce((n, row) => n + (row.q[i] - mean) ** 2, 0) / lastStats.length);
          });
          $('preview-jitter').textContent = `Target variation (2 s): ${Math.max(...std).toFixed(3)} max σ · includes intentional motion`;
        }
      });
      function animate(now) {
        if (playback?.playing) {
          // Bound catch-up work; if the tab stalled, pause rather than omit filter samples.
          const due = playback.tick(now);
          if (due.length > 10 || !hooks.connected()) { pause(); playback.rewind(); stream++; $('clip-progress').value = 0; message('Replay reset after a timing/connection interruption. Press replay to repeat from the start.'); }
          else for (const {frame, index} of due) {
            hooks.replay({timestamp: frame.t, hands: frame.hands, inference_ms: frame.inference_ms}, {
              source: 'replay', stream: `replay-${pageId}-${stream}`, index, swap: clip.swap,
              profile: $('clip-profile').checked ? clip.profile : undefined,
            });
            $('clip-progress').value = index; $('clip-progress').max = clip.frames.length - 1;
          }
          if (!playback.playing) {
            $('clip-play').textContent = 'Replay in preview';
            if (playback.index === playback.clip.frames.length) message('Replay complete · preview held at the final frame');
          }
        }
        requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
    },
    observe(data) {
      if (recorder && !recorder.add(data)) stopRecording();
    },
    stopRecording,
    pause,
    getClip: () => clip,
    leaveReview,
    reviewFrame() {
      const index = playback ? playback.index - 1 : -1;
      const shown = hooks.state()?.preview;
      if (!clip || !playback || playback.playing || index < 0 || shown?.source !== 'replay'
          || shown.seq !== index || shown.stream !== `replay-${pageId}-${stream}`)
        throw new Error('Replay a clip, pause it, then wait for that frame to appear before editing.');
      return {index, target: shown.target.slice()};
    },
    step() {
      stopRecording(); pause(); leaveReview();
      if (!playback) { message('Load or record a clip first.'); return; }
      hooks.stopCamera();
      if (playback.index === playback.clip.frames.length) { message('End of clip; rewind to start again.'); return; }
      if (playback.index === 0) stream++;
      const index = playback.index++, frame = clip.frames[index];
      hooks.replay({timestamp: frame.t, hands: frame.hands, inference_ms: frame.inference_ms}, {
        source: 'replay', stream: `replay-${pageId}-${stream}`, index, swap: clip.swap,
        profile: $('clip-profile').checked ? clip.profile : undefined,
      });
      playback.elapsed = frame.t;
      $('clip-progress').value = index; $('clip-progress').max = clip.frames.length - 1;
      message(`Paused at frame ${index} · ${(frame.t / 1000).toFixed(2)} s`);
    },
    isReplaying: () => !!playback?.playing,
  };
})();
