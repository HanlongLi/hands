import * as THREE from './vendor/three/three.module.js';
import {STLLoader} from './vendor/three/STLLoader.js';
import {OrbitControls} from './vendor/three/OrbitControls.js';

const host = document.getElementById('hand-preview');
const status = document.getElementById('preview-model-status');
async function init() {
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', 'Interactive 3D Revo2 right hand');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 10);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 0.1; controls.maxDistance = 1.5;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x687890, 2.5));
  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.position.set(0.2, -0.3, 0.5); scene.add(light);
  const response = await fetch('/preview/model');
  if (!response.ok) throw new Error('Revo2 model is unavailable');
  const model = await response.json();
  const geometries = new Map(), loader = new STLLoader();
  async function geometry(url) {
    if (!geometries.has(url)) geometries.set(url, loader.loadAsync(url));
    return geometries.get(url);
  }
  // URDF origin transforms are Rz(yaw) Ry(pitch) Rx(roll).
  function origin(object, data) {
    object.position.fromArray(data.xyz);
    object.rotation.set(...data.rpy, 'ZYX');
  }
  async function makeHand(color, wireframe = false) {
    const links = Object.fromEntries(model.links.map(link => [link.name, new THREE.Group()]));
    const moving = new Map();
    const material = new THREE.MeshStandardMaterial({color, metalness: 0.15,
      roughness: 0.55, wireframe, transparent: wireframe, opacity: wireframe ? 0.28 : 1});
    await Promise.all(model.links.flatMap(link => link.visuals.map(async visual => {
      const mesh = new THREE.Mesh(await geometry(visual.mesh), material);
      origin(mesh, visual); mesh.scale.fromArray(visual.scale); links[link.name].add(mesh);
    })));
    for (const joint of model.joints) {
      const fixed = new THREE.Group(), motion = new THREE.Group();
      origin(fixed, joint); links[joint.parent].add(fixed); fixed.add(motion);
      motion.add(links[joint.child]); moving.set(joint.name, motion);
    }
    const root = links[model.root]; scene.add(root);
    function update(values) {
      const angles = Object.fromEntries(model.drives.map((drive, i) => [drive.joint, values[i] * drive.range]));
      for (const joint of model.joints) {
        if (joint.mimic) angles[joint.name] = (angles[joint.mimic.joint] || 0) * Number(joint.mimic.multiplier || 1) + Number(joint.mimic.offset || 0);
        if (joint.type === 'revolute') moving.get(joint.name).quaternion.setFromAxisAngle(
          new THREE.Vector3(...joint.axis).normalize(), angles[joint.name] || 0);
      }
    }
    return {root, update};
  }
  const mapped = await makeHand(0x40b8ac);
  const measured = await makeHand(0xeea64b, true);
  mapped.update([0, 0, 0, 0, 0, 0]);
  const center = new THREE.Box3().setFromObject(mapped.root).getCenter(new THREE.Vector3());
  function resetView() {
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(0.18, -0.24, 0.12));
    controls.update();
  }
  resetView();
  document.getElementById('preview-reset').onclick = resetView;
  const resize = new ResizeObserver(() => {
    const width = host.clientWidth, height = host.clientHeight;
    renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
  });
  resize.observe(host);
  status.textContent = 'Revo2 model ready · drag to rotate · scroll to zoom';
  let target = [0, 0, 0, 0, 0, 0], actual = target, reviewTarget = null, reviewFrame = null;
  window.addEventListener('r1hands-label', ({detail}) => {
    reviewTarget = detail.position; reviewFrame = detail.frame;
  });
  window.addEventListener('r1hands-review-reset', () => { reviewTarget = null; reviewFrame = null; });
  window.addEventListener('r1hands-state', ({detail}) => {
    const preview = detail.preview;
    const mode = document.getElementById('preview-mode').value;
    const useMapping = mode === 'mapping';
    target = useMapping && preview ? preview.target : (detail.hands.right?.target || target);
    if (mode === 'review' && reviewTarget) target = reviewTarget;
    actual = detail.hands.right?.position || target;
    document.getElementById('preview-values').textContent = target.map((v, i) => `${['thumb', 'rotation', 'index', 'middle', 'ring', 'pinky'][i]} ${v.toFixed(2)}`).join(' · ');
    document.getElementById('preview-state').textContent = useMapping && preview
      ? `${preview.source} · ${preview.status} · ${preview.mapper || 'analytic v1'} · ${preview.mapper?.startsWith('learned') ? 'calibration handled by trained weights' : (preview.calibration === 'recorded' ? 'recorded calibration' : (detail.calibrated.right ? 'current calibration' : 'default calibration'))}`
      : `panel target · ${detail.conn.backend}`;
    if (mode === 'review' && reviewTarget) document.getElementById('preview-state').textContent = `Operator label · frame ${reviewFrame} · preview only`;
    renderer.domElement.dataset.pose = JSON.stringify(target);
  });
  function draw() {
    mapped.update(target); measured.update(actual);
    measured.root.visible = document.getElementById('preview-measured').checked;
    controls.update(); renderer.render(scene, camera); requestAnimationFrame(draw);
  }
  draw();
  host.dataset.ready = 'true';
}
init().catch(error => {
  status.textContent = `3D preview unavailable: ${error.message}. Try Chrome with graphics acceleration enabled.`;
  host.dataset.ready = 'error';
});
