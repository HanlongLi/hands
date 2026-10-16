// Lightweight contract test for the camera controller, without a camera or downloaded model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/r1hands/panel/static/index.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    checked: false, textContent: '', lastElementChild: {textContent: ''}, classList: {toggle(){}, add(){}, remove(){}},
    addEventListener(){}, getContext: () => ({beginPath(){}, arc(){}, fill(){}, save(){},
      translate(){}, scale(){}, fillText(){}, restore(){}}),
    videoWidth: 640, videoHeight: 480,
  });
  return elements.get(id);
}
let sent = [], now = 1000;
class Socket {
  constructor(){ this.readyState = 1; this.bufferedAmount = 0; }
  send(value){ sent.push(JSON.parse(value)); }
}
const context = vm.createContext({console, WebSocket: Socket,
  document: {getElementById: element, addEventListener(){}, hidden: false},
  window: {addEventListener(){}}, location: {host: 'localhost:8080'},
  performance: {now: () => now}, requestAnimationFrame(){}, setInterval(){}, setTimeout(){},
  getComputedStyle: () => ({getPropertyValue: () => '#000'}), alert(){},
});
vm.runInContext(source, context);
vm.runInContext('camOn = true; drivingTeleop = true; S = {estopped: false, panel_time: 123};', context);
const hand = {label:'left', score:0.99, points:Array.from({length:21},()=>[0,0,0]),
  image:Array.from({length:21},()=>({x:0.5,y:0.5}))};
context.result = {timestamp:990, inference_ms:8, hands:[hand]};
vm.runInContext('acceptTracking(result)', context);
assert.equal(sent.at(-1).type, 'teleop');
assert.deepEqual(Object.keys(sent.at(-1).hands), ['right']);
assert.equal(sent.at(-1).age_ms, 10);
const count = sent.length;
now = 1300;
vm.runInContext('acceptTracking(result)', context);
assert.equal(sent.length, count, 'stale inference must be dropped');
now = 1000;
vm.runInContext('ws.bufferedAmount = 100; acceptTracking(result)', context);
assert.equal(sent.length, count, 'backpressure must drop frames');
vm.runInContext('ws.bufferedAmount = 0', context);
context.result = {timestamp:990, inference_ms:8, hands:[{...hand, score:0.2}]};
vm.runInContext('acceptTracking(result)', context);
assert.deepEqual(sent.at(-1).hands, {}, 'uncertain identity must not move a hand');
context.result = {timestamp:990, inference_ms:8, hands:[hand, hand]};
vm.runInContext('acceptTracking(result)', context);
assert.deepEqual(sent.at(-1).hands, {}, 'duplicate anatomical labels are ambiguous');
vm.runInContext('ws.onclose()', context);
assert.equal(vm.runInContext('drivingTeleop', context), false);
console.log('Webcam UI contracts passed');
