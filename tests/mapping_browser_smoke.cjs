// Real-browser mapping annotation/export regression. Uses only the mock panel.
const {chromium} = require('playwright-core');
const fs = require('node:fs/promises');
const {execFileSync} = require('node:child_process');
const assert = require('node:assert/strict');
const base = process.env.R1H_PANEL_URL || 'http://127.0.0.1:8098';
(async () => {
  const browser = await chromium.launch({channel:'chrome',headless:true});
  try {
    const page = await browser.newPage();
    const errors = [], sent = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('websocket', socket => socket.on('framesent', ({payload}) => sent.push(JSON.parse(payload))));
    await page.goto(base);
    await page.waitForFunction(() => document.getElementById('hand-preview').dataset.ready === 'true', {}, {timeout:30000});
    await page.locator('#clip-example').click();
    await page.waitForFunction(() => document.getElementById('clip-status').textContent.includes('180 frames'));
    await page.locator('#mapping-lab summary').click();
    await page.locator('#mapping-step').click();
    await page.waitForFunction(() => { try { return DemoLab.reviewFrame().index === 0; } catch { return false; } });
    await page.locator('#mapping-edit').click();
    assert.ok((await page.locator('#mapping-frame').textContent()).startsWith('Frame 0'));
    await page.locator('#mapping-axis-2').focus();
    await page.locator('#mapping-axis-2').press('End');
    await page.locator('#mapping-reviewer').fill('browser-tester');
    await page.locator('#mapping-note').fill('index target adjusted');
    await page.locator('#mapping-save').click();
    await page.waitForFunction(() => document.getElementById('mapping-status').textContent.includes('1 reviewed'));
    await page.waitForFunction(() => JSON.parse(document.querySelector('#hand-preview canvas').dataset.pose)[2] === 1);
    const jsonDownload = page.waitForEvent('download');
    await page.locator('#clip-download').click();
    const clip = JSON.parse(await fs.readFile(await (await jsonDownload).path(), 'utf8'));
    assert.equal(clip.annotations[0].frame_index, 0);
    assert.equal(clip.annotations[0].position[2], 1);
    assert.equal(clip.annotations[0].reviewer, 'browser-tester');
    // Stepping invalidates the editor: a label must never attach to a stale frame selection.
    await page.locator('#mapping-step').click();
    assert.equal(await page.locator('#mapping-save').isDisabled(), true);
    await page.locator('#clip-load').setInputFiles({name:'reviewed.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(clip))});
    await page.waitForFunction(() => DemoLab.getClip()?.annotations?.length === 1);
    const datasetDownload = page.waitForEvent('download');
    await page.locator('#mapping-export').click();
    const npz = await (await datasetDownload).path();
    const summary = JSON.parse(execFileSync('.venv/bin/python', ['-c',
      'import numpy as np,json,sys; d=np.load(sys.argv[1],allow_pickle=False); print(json.dumps({"labels":int(d["label_valid"].sum()),"samples":int(d["training_mask"].sum()),"target":float(d["corrected_robot_target"][0,2]),"collision_checked":bool(d["collision_checked"].any())}))',npz], {encoding:'utf8'}));
    assert.equal(summary.labels,1); assert.equal(summary.samples,1); assert.equal(summary.target,1);
    assert.equal(summary.collision_checked,false);
    assert.ok(!sent.some(m => ['teleop','finger','grasp','routine'].includes(m.type)), 'annotation/export must not send motor targets');
    assert.deepEqual(errors, []);
    console.log('Paused-frame annotation, 3D label editing, clip roundtrip, training export and no-motor-command contracts passed');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
