// Portable landmark clips, independent of camera tracking, 3D rendering, and motor control.
(function (root) {
  'use strict';
  const MAX_FRAMES = 3600, MAX_DURATION = 120000, MAX_BYTES = 18 * 1024 * 1024;
  function calibration(value) {
    if (!value || !['open', 'closed'].every(key => Array.isArray(value[key]) && value[key].length === 6 && value[key].every(Number.isFinite)))
      throw new Error('Clip needs a six-axis calibration');
    for (let i = 0; i < 6; i++) if (Math.abs(value.closed[i] - value.open[i]) < 0.01 || Math.abs(value.open[i]) > 10 || Math.abs(value.closed[i]) > 10)
      throw new Error('Clip calibration range is invalid');
  }
  function validate(clip) {
    if (clip?.schema !== 'r1hands.landmarks.v1' || clip.side !== 'right' || typeof clip.swap !== 'boolean')
      throw new Error('Expected a right-hand r1hands landmark clip (version 1)');
    calibration(clip.profile);
    if (!Array.isArray(clip.frames) || !clip.frames.length || clip.frames.length > MAX_FRAMES)
      throw new Error('Clip must contain 1–3600 frames');
    let previous = -1;
    for (const frame of clip.frames) {
      if (!Number.isFinite(frame.t) || frame.t < 0 || frame.t <= previous || frame.t > MAX_DURATION)
        throw new Error('Frame times must increase and remain within two minutes');
      previous = frame.t;
      if (!Array.isArray(frame.hands) || frame.hands.length > 2) throw new Error('Invalid hand observations');
      for (const hand of frame.hands) {
        if (!['left', 'right'].includes(hand.label) || !Number.isFinite(hand.score) || hand.score < 0 || hand.score > 1)
          throw new Error('Invalid hand identity');
        if (!Array.isArray(hand.points) || hand.points.length !== 21 || !hand.points.every(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)))
          throw new Error('Expected 21 finite 3D landmarks');
        if (!Array.isArray(hand.image) || hand.image.length !== 21 || !hand.image.every(p => p && Number.isFinite(p.x) && Number.isFinite(p.y)))
          throw new Error('Expected 21 image landmarks');
      }
    }
    return clip;
  }
  class Recorder {
    constructor(profile, swap) {
      calibration(profile);
      this.clip = {schema: 'r1hands.landmarks.v1', side: 'right', synthetic: false,
        created_at: new Date().toISOString(), swap, profile: JSON.parse(JSON.stringify(profile)), frames: []};
      this.first = null; this.bytes = JSON.stringify(this.clip).length;
    }
    add(data) {
      if (this.first === null) this.first = data.timestamp;
      const t = data.timestamp - this.first;
      const frames = this.clip.frames;
      if (frames.length >= MAX_FRAMES || t > MAX_DURATION) return false;
      if (frames.length && t <= frames.at(-1).t) return true;
      const frame = {t, hands: JSON.parse(JSON.stringify(data.hands)), inference_ms: data.inference_ms};
      const bytes = JSON.stringify(frame).length + 1;
      if (this.bytes + bytes > MAX_BYTES) return false;
      this.bytes += bytes; frames.push(frame);
      return true;
    }
  }
  // Scheduler advances on a caller-provided clock. No timers, I/O, or motor commands.
  class Playback {
    constructor(clip) { this.clip = validate(clip); this.index = 0; this.elapsed = 0; this.playing = false; }
    play(now) { this.origin = now - this.elapsed; this.playing = true; }
    pause(now) { if (this.playing) this.elapsed = Math.max(0, now - this.origin); this.playing = false; }
    rewind() { this.index = 0; this.elapsed = 0; this.playing = false; }
    tick(now) {
      if (!this.playing) return [];
      this.elapsed = Math.max(0, now - this.origin);
      const due = [];
      while (this.index < this.clip.frames.length && this.clip.frames[this.index].t <= this.elapsed) {
        due.push({frame: this.clip.frames[this.index], index: this.index}); this.index++;
      }
      if (this.index === this.clip.frames.length) this.playing = false;
      return due;
    }
  }
  root.LandmarkClip = {validate, Recorder, Playback};
})(globalThis);
