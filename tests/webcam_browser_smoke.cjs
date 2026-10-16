// Optional real-browser regression. Run against a mock panel; uses a synthetic camera.
// NODE_PATH=/tmp/r1hands-browser-test/node_modules node tests/webcam_browser_smoke.cjs
const {chromium} = require('playwright-core');
const assert = require('node:assert/strict');
const url = process.env.R1H_PANEL_URL || 'http://127.0.0.1:8097';
(async () => {
  const browser = await chromium.launch({channel: 'chrome', headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream']});
  try {
    const page = await browser.newPage();
    const errors = [], dialogs = [], externalRequests = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('dialog', async dialog => { dialogs.push(dialog.message()); await dialog.dismiss(); });
    await page.route('**/*', route => {
      const target = route.request().url();
      if (!target.startsWith(url + '/') && !target.startsWith('blob:')) {
        externalRequests.push(target);
        return route.abort();
      }
      return route.continue();
    });
    await page.goto(url);
    await page.evaluate(() => {
      window.trackingFrames = 0;
      const original = acceptTracking;
      acceptTracking = data => { window.trackingFrames++; original(data); };
    });
    await page.getByRole('button', {name: 'Start camera', exact: true}).click();
    await page.waitForFunction(() => window.trackingFrames >= 3, {}, {timeout: 30000});
    assert.equal(await page.locator('#cam').textContent(), 'Stop camera');
    assert.deepEqual(errors, []);
    assert.deepEqual(dialogs, []);
    assert.deepEqual(externalRequests, [], 'runtime must not depend on CDN or model downloads');
    await page.getByRole('button', {name: 'Stop camera', exact: true}).click();
    assert.equal(await page.locator('#cam').textContent(), 'Start camera');
    console.log('Chrome model initialization and three inference frames passed with external requests blocked');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
