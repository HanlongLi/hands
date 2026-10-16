// Run against an isolated mock panel with .[panel,sim] installed.
const {chromium} = require('playwright-core');
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
    await page.waitForFunction(() => typeof S !== 'undefined' && !!S);
    await page.evaluate(() => { if (S.estopped) send({type:'resume'}); });
    await page.waitForFunction(() => !S.estopped);
    assert.equal(await page.locator('#panel-mapping').isVisible(),true);
    await page.locator('#tab-task').click();
    assert.equal(await page.locator('#panel-mapping').isVisible(),false);
    assert.equal(await page.locator('#tab-task').getAttribute('aria-selected'),'true');
    if (await page.locator('#task-start').isEnabled()) await page.locator('#task-start').click();
    await page.waitForFunction(() => !document.getElementById('task-reset').disabled, {}, {timeout:35000});
    await page.locator('#task-seed').fill('0');
    await page.locator('#task-reset').click();
    await page.waitForFunction(() => document.getElementById('task-status').textContent === 'Support in place');
    await page.waitForFunction(() => document.getElementById('task-image').naturalWidth === 320);
    const previousDownload=await page.locator('#task-saved a').evaluateAll(links => links[0]?.getAttribute('href') || null);
    await page.locator('#task-operator').fill('scripted-browser-regression');
    await page.locator('#task-begin').click();
    await page.waitForFunction(() => !document.getElementById('task-finish').disabled);
    assert.equal(await page.locator('#task-reset').isDisabled(),true);
    await page.locator('#task-example').click();
    await page.waitForTimeout(1500);
    await page.locator('#task-release').click();
    await page.waitForFunction(previous => {
      const link=document.querySelector('#task-saved a');
      return link && link.getAttribute('href')!==previous && link.textContent.includes('success');
    }, previousDownload, {timeout:10000});
    assert.equal(await page.locator('#task-finish').isDisabled(),true);
    const downloadPromise=page.waitForEvent('download');
    await page.locator('#task-saved a').click();
    const filename=await (await downloadPromise).path();
    const summary=JSON.parse(execFileSync('.venv/bin/python',['-c',
      'import numpy as np,json,sys; d=np.load(sys.argv[1],allow_pickle=False); m=json.loads(str(d["metadata_json"])); print(json.dumps({"success":m["success"],"hardware":m["hardware"],"sources":m["source_counts"],"states":len(d["observation_state"]),"actions":len(d["action"]),"rgb":list(d["observation_rgb"].shape)}))',filename],{encoding:'utf8'}));
    assert.equal(summary.success,true);assert.equal(summary.hardware,false);
    assert.ok(summary.sources.scripted>0);
    assert.equal(summary.states,summary.actions+1);
    assert.deepEqual(summary.rgb,[summary.states,240,320,3]);
    // Tab keyboard access and a narrow viewport must keep controls usable.
    await page.locator('#tab-task').focus();await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#panel-logs').isVisible(),true);
    await page.locator('#tab-task').click();
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:'/tmp/r1hands-task-mobile.png',fullPage:true});
    assert.ok(!sent.some(m => ['teleop','finger','grasp','routine'].includes(m.type)),'task controls must not send robot targets');
    assert.deepEqual(errors,[]);
    console.log('Task tabs, MuJoCo RGB, scripted grasp success, aligned episode download, mobile layout and no-robot-command checks passed');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
