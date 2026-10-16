// Run with a mock panel on R1H_PANEL_URL (default :8097), Chrome and playwright-core.
const {chromium} = require('playwright-core');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const base = process.env.R1H_PANEL_URL || 'http://127.0.0.1:8097';
(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [], external = [], sent = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('dialog', async dialog => { errors.push(dialog.message()); await dialog.dismiss(); });
    page.on('websocket', socket => socket.on('framesent', ({payload}) => sent.push(JSON.parse(payload))));
    await page.route('**/*', route => {
      if (!route.request().url().startsWith(base + '/') && !route.request().url().startsWith('blob:')) {
        external.push(route.request().url()); return route.abort();
      }
      return route.continue();
    });
    await page.goto(base);
    await page.waitForFunction(() => document.getElementById('hand-preview').dataset.ready === 'true', {}, {timeout: 30000});
    assert.equal(await page.locator('#hand-preview canvas').count(), 1);
    await page.locator('#clip-example').click();
    await page.waitForFunction(() => document.getElementById('clip-status').textContent.includes('180 frames'));
    await page.locator('#clip-play').click();
    await page.waitForFunction(() => {
      const pose = document.querySelector('#hand-preview canvas').dataset.pose;
      return Number(document.getElementById('clip-progress').value) > 10
        && document.getElementById('preview-state').textContent.includes('replay')
        && pose && JSON.parse(pose)[2] > 0.7;
    });
    assert.ok((await page.locator('#preview-state').textContent()).includes('replay'));
    await page.locator('#clip-play').click(); // pause
    const paused = await page.locator('#clip-progress').getAttribute('value');
    await page.waitForTimeout(150);
    assert.equal(await page.locator('#clip-progress').getAttribute('value'), paused);
    await page.locator('#clip-rewind').click();
    assert.equal(await page.locator('#clip-progress').getAttribute('value'), '0');
    assert.ok(!sent.some(m => ['teleop', 'finger', 'grasp', 'routine'].includes(m.type)), 'replay must not command motors');
    const fileEvent = page.waitForEvent('download');
    await page.locator('#clip-download').click();
    const download = await fileEvent;
    const data = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
    assert.equal(data.schema, 'r1hands.landmarks.v1');
    assert.equal(data.frames.length, 180);
    await page.locator('#clip-load').setInputFiles({name:'roundtrip.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(data))});
    await page.waitForFunction(() => document.getElementById('clip-status').textContent.includes('180 frames'));
    await page.locator('#clip-load').setInputFiles({name:'invalid.json',mimeType:'application/json',buffer:Buffer.from('{"schema":"bad"}')});
    await page.waitForFunction(() => document.getElementById('clip-status').textContent.includes('Could not load clip'));
    // The real worker still loads and records with an artificial camera after the preview was added.
    await page.getByRole('button', {name:'Start camera',exact:true}).click();
    await page.waitForFunction(() => document.getElementById('cam').textContent === 'Stop camera', {}, {timeout:30000});
    await page.locator('#landmark-record').click();
    await page.waitForTimeout(700);
    await page.locator('#landmark-record').click();
    assert.ok((await page.locator('#clip-status').textContent()).includes('frames'));
    const recordedDownload = page.waitForEvent('download');
    await page.locator('#clip-download').click();
    const recorded = JSON.parse(await fs.readFile(await (await recordedDownload).path(), 'utf8'));
    assert.ok(recorded.frames.length >= 3);
    assert.equal(recorded.synthetic, false);
    await page.getByRole('button',{name:'Stop camera',exact:true}).click();
    await page.setViewportSize({width:390,height:844});
    await page.locator('#teleop').scrollIntoViewIfNeeded();
    await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth + 1, {}, {timeout:3000});
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    console.log('3D model, offline preview replay, pause/rewind, import/download, real camera recording and mobile layout passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
