// Operator labels and offline data export. This module never sends motor commands.
(function () {
  const $ = id => document.getElementById(id);
  let selected = null, selectedClip = null, target = [0, 0, 0, 0, 0, 0];
  const names = ['Thumb', 'Thumb rot.', 'Index', 'Middle', 'Ring', 'Pinky'];
  const status = text => { $('mapping-status').textContent = text; };
  $('mapping-sliders').innerHTML = names.map((name, i) => `<label class="mapping-axis">${name}
    <input id="mapping-axis-${i}" type="range" min="0" max="1" step="0.01" value="0" disabled aria-label="Reviewed ${name}">
    <output id="mapping-value-${i}">0.00</output></label>`).join('');
  function draw() {
    for (let i = 0; i < 6; i++) {
      $(`mapping-axis-${i}`).value = target[i]; $(`mapping-value-${i}`).textContent = target[i].toFixed(2);
    }
    $('preview-mode').value = 'review';
    window.dispatchEvent(new CustomEvent('r1hands-label', {detail: {position: target.slice(), frame: selected}}));
  }
  function validSelection() {
    try { return selected !== null && selectedClip === DemoLab.getClip() && DemoLab.reviewFrame().index === selected; }
    catch { return false; }
  }
  for (let i = 0; i < 6; i++) $(`mapping-axis-${i}`).oninput = event => {
    if (!validSelection()) { status('Select a paused pose again before editing.'); return; }
    target[i] = Number(event.target.value); draw();
  };
  window.addEventListener('r1hands-review-reset', () => {
    selected = null; selectedClip = null;
    $('mapping-save').disabled = true; $('mapping-remove').disabled = true;
    for (let i = 0; i < 6; i++) $(`mapping-axis-${i}`).disabled = true;
    $('mapping-frame').textContent = 'No pose selected';
  });
  $('mapping-step').onclick = () => { DemoLab.step(); };
  $('mapping-edit').onclick = () => {
    try {
      const frame = DemoLab.reviewFrame();
      selected = frame.index; selectedClip = DemoLab.getClip();
      const existing = selectedClip.annotations?.find(a => a.frame_index === selected);
      target = (existing?.position || frame.target).slice();
      $('mapping-note').value = existing?.note || '';
      $('mapping-frame').textContent = `Frame ${selected} · ${(selectedClip.frames[selected].t / 1000).toFixed(2)} s`;
      for (let i = 0; i < 6; i++) $(`mapping-axis-${i}`).disabled = false;
      $('mapping-save').disabled = false; $('mapping-remove').disabled = false;
      draw(); status('Adjust the preview, then save this pose. Motor targets are unchanged.');
    } catch (error) { status(error.message); }
  };
  $('mapping-save').onclick = () => {
    if (!validSelection()) { status('Select a paused pose again before saving.'); return; }
    const annotations = selectedClip.annotations || [];
    selectedClip.annotations = annotations.filter(a => a.frame_index !== selected);
    selectedClip.annotations.push({frame_index: selected, position: target.slice(),
      reviewer: $('mapping-reviewer').value.trim(), note: $('mapping-note').value.trim()});
    status(`Saved frame ${selected} · ${selectedClip.annotations.length} reviewed pose(s). Download the clip to keep labels; export the dataset for training.`);
  };
  $('mapping-remove').onclick = () => {
    if (!validSelection()) return;
    selectedClip.annotations = (selectedClip.annotations || []).filter(a => a.frame_index !== selected);
    status(`Removed frame ${selected} label · ${selectedClip.annotations.length} reviewed pose(s) remain.`);
  };
  $('mapping-export').onclick = async () => {
    DemoLab.stopRecording(); DemoLab.pause();
    const clip = DemoLab.getClip();
    if (!clip) { status('Record or import a clip first.'); return; }
    $('mapping-export').disabled = true;
    try {
      LandmarkClip.validate(clip);
      status('Building synchronized mapping samples…');
      const response = await fetch('/mapping/export', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(clip)});
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob(), url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = `right-hand-mapping-${Date.now()}.npz`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      status('Mapping dataset downloaded. Unlabeled frames remain masked; operator labels are not collision-validated.');
    } catch (error) { status(`Export failed: ${error.message}`); }
    finally { $('mapping-export').disabled = false; }
  };
})();
