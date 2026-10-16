"""Independent fixed-wrist MuJoCo task. No HAL or real-robot connections."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from r1hands.common.robot_model import REPO
from r1hands.common.types import HAND_JOINTS
from r1hands.sim.build_model import HAND_DRIVE, HAND_EFFORT, HAND_RANGE

DESCRIPTION = REPO / 'assets/revo2_description'
TASK_VERSION = 'revo2-supported-block-v1'
STATE_NAMES = ([f'hand_q_{j}' for j in HAND_JOINTS] + [f'hand_dq_{j}' for j in HAND_JOINTS]
               + ['object_x', 'object_y', 'object_z', 'object_qw', 'object_qx', 'object_qy', 'object_qz']
               + ['object_vx', 'object_vy', 'object_vz', 'object_wx', 'object_wy', 'object_wz', 'support_removed'])


@dataclass(frozen=True)
class TaskConfig:
    control_dt: float = 0.05
    hold_s: float = 1.0
    force_limit_n: float = 25.0
    displacement_limit_m: float = 0.025
    speed_limit_m_s: float = 0.04
    minimum_object_height_m: float = 0.16
    max_steps: int = 400
    width: int = 320
    height: int = 240


@lru_cache
def model_xml():
    """URDF dynamics/kinematics with explicit primitive contact geometry."""
    import mujoco
    root = ET.parse(DESCRIPTION / 'urdf/revo2_right_hand.urdf').getroot()
    for mesh in root.iter('mesh'):
        mesh.set('filename', str(DESCRIPTION / mesh.get('filename').split('package://revo2_description/')[1]))
    mj = ET.SubElement(root, 'mujoco')
    ET.SubElement(mj, 'compiler', fusestatic='false', balanceinertia='true', discardvisual='true')
    imported = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    with tempfile.TemporaryDirectory() as directory:
        filename = str(Path(directory) / 'hand.xml')
        mujoco.mj_saveLastXML(filename, imported)
        model = ET.parse(filename).getroot()
    model.set('model', TASK_VERSION)
    ET.SubElement(model, 'option', timestep='0.002', integrator='implicitfast', gravity='0 0 -9.81')
    world = model.find('worldbody')
    base = world.find('body')
    base.set('pos', '0 0 0.18')
    for body in model.iter('body'):
        name = body.get('name', '')
        for geom in body.findall('geom'):
            geom.set('contype', '0'); geom.set('conaffinity', '0'); geom.set('group', '2')
            geom.set('rgba', '0.15 0.66 0.61 1')
        for joint in body.findall('joint'):
            joint.set('armature', '0.002'); joint.set('damping', '0.02')
        if name == 'right_base_link':
            ET.SubElement(body, 'geom', name='palm_contact', type='box', pos='-0.006 0 0.043',
                          size='0.013 0.033 0.032', contype='2', conaffinity='1', group='3', rgba='0.1 0.7 0.4 0.2')
        elif name.endswith(('_proximal_link', '_distal_link', '_metacarpal_link')):
            endpoint = None
            for child in body.findall('body'):
                if child.get('name', '').endswith(('_distal_link', '_proximal_link', '_tip_link')):
                    endpoint = child.get('pos'); break
            if endpoint:
                ET.SubElement(body, 'geom', name=name + '_contact', type='capsule',
                              fromto='0 0 0 ' + endpoint, size='0.007', friction='1.2 0.01 0.001',
                              contype='2', conaffinity='1', group='3', rgba='0.1 0.7 0.4 0.2')
    # Hand contact bit 2 collides with object/world bit 1. Self-contact remains outside
    # this first task's validated scope; the primitive shapes avoid palm convex-hull artifacts.
    equality = model.find('equality')
    if equality is None:
        equality = ET.SubElement(model, 'equality')
    pairs = {e.get('joint1') for e in equality.findall('joint')}
    for joint in root.findall('joint'):
        mimic = joint.find('mimic')
        if mimic is not None and joint.get('name') not in pairs:
            ET.SubElement(equality, 'joint', joint1=joint.get('name'), joint2=mimic.get('joint'),
                          polycoef=f"0 {mimic.get('multiplier', '1')} 0 0 0", solref='0.005 1')
    actuators = ET.SubElement(model, 'actuator')
    for name in HAND_JOINTS:
        effort = HAND_EFFORT[name]
        ET.SubElement(actuators, 'position', name=name, joint='right_' + HAND_DRIVE[name],
                      kp=str(effort / 0.25), kv=str(effort / 20), ctrlrange=f'0 {HAND_RANGE[name]}',
                      forcerange=f'{-effort} {effort}', forcelimited='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0.5 0.5 0.01', rgba='0.82 0.84 0.85 1')
    support = ET.SubElement(world, 'body', name='support', mocap='true', pos='0.04 0 0.262')
    ET.SubElement(support, 'geom', name='support_geom', type='box', size='0.012 0.012 0.003', contype='4', conaffinity='1', rgba='0.4 0.47 0.55 1')
    obj = ET.SubElement(world, 'body', name='block', pos='0.04 0 0.285')
    ET.SubElement(obj, 'freejoint', name='object_free')
    ET.SubElement(obj, 'geom', name='block_geom', type='box', size='0.02 0.02 0.02', mass='0.04',
                  friction='1.2 0.01 0.001', rgba='0.94 0.51 0.16 1', condim='4')
    ET.SubElement(world, 'light', pos='0.4 -0.3 0.8', dir='-0.4 0.3 -0.8')
    ET.SubElement(world, 'camera', name='task', pos='0.34 -0.42 0.4', xyaxes='0.78 0.63 0 -0.2 0.25 0.947')
    return ET.tostring(model, encoding='unicode')


class HandTask:
    def __init__(self, config=None):
        import mujoco
        self.mj = mujoco
        self.config = config or TaskConfig()
        self.xml = model_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint('right_' + HAND_DRIVE[j]) for j in HAND_JOINTS]
        self.qadr = [j.qposadr[0] for j in self.joints]
        self.vadr = [j.dofadr[0] for j in self.joints]
        self.ranges = np.array([HAND_RANGE[j] for j in HAND_JOINTS])
        self.object_q = self.model.joint('object_free').qposadr[0]
        self.object_v = self.model.joint('object_free').dofadr[0]
        self.object_geom = self.model.geom('block_geom').id
        self.support_mocap = self.model.body('support').mocapid[0]
        self.renderer = None
        self.reset(0)

    def reset(self, seed=0):
        self.mj.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.initial_position = np.array([0.04 + rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004), 0.285])
        self.data.qpos[self.object_q:self.object_q+3] = self.initial_position
        self.data.qpos[self.object_q+3:self.object_q+7] = [1, 0, 0, 0]
        self.data.mocap_pos[self.support_mocap] = [self.initial_position[0], self.initial_position[1], 0.262]
        self.target = np.zeros(6)
        self.support_removed = False
        self.hold_time = 0.0
        self.hold_reference = None
        self.success = False
        self.steps = 0
        self.mj.mj_forward(self.model, self.data)
        return self.observe()

    def release_support(self):
        self.support_removed = True
        self.data.mocap_pos[self.support_mocap, 2] = -0.1
        self.mj.mj_forward(self.model, self.data)

    def contacts(self):
        groups, normals, maximum, external = set(), {}, 0.0, False
        for contact_index, contact in enumerate(self.data.contact):
            if self.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            name = self.model.body(self.model.geom_bodyid[other]).name
            force = np.zeros(6)
            self.mj.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = abs(float(force[0]))
            maximum = max(maximum, normal_force)
            group = next((finger for finger in ('thumb','index','middle','ring','pinky') if f'_{finger}_' in name), None)
            if group and normal_force > 0.03:
                groups.add(group)
                direction = np.array(contact.frame[:3]) * (1 if contact.geom1 == self.object_geom else -1)
                normals.setdefault(group, []).append(direction)
            if name in ('world', 'support') and normal_force > 0.03:
                external = True
        opposing = any(np.dot(a,b) < -0.2 for a in normals.get('thumb', [])
                       for key, values in normals.items() if key != 'thumb' for b in values)
        return {'fingers': sorted(groups), 'max_force_n': maximum, 'opposing': opposing, 'external_support': external}

    def observe(self):
        d = self.data
        q = d.qpos[self.qadr] / self.ranges
        dq = d.qvel[self.vadr] / self.ranges
        obj_q = d.qpos[self.object_q:self.object_q+7].copy()
        obj_v = d.qvel[self.object_v:self.object_v+6].copy()
        state = np.r_[q, dq, obj_q, obj_v, float(self.support_removed)]
        contact = self.contacts()
        stable = (self.support_removed and contact['opposing'] and not contact['external_support']
                  and contact['max_force_n'] <= self.config.force_limit_n
                  and obj_q[2] >= self.config.minimum_object_height_m
                  and (self.hold_reference is None or np.linalg.norm(obj_q[:3] - self.hold_reference) <= self.config.displacement_limit_m)
                  and np.linalg.norm(obj_v[:3]) <= self.config.speed_limit_m_s)
        return {'time_s': float(d.time), 'state': state, 'position': q.copy(), 'target': self.target.copy(),
                'object_position': obj_q[:3].copy(), 'support_removed': self.support_removed,
                'contacts': contact, 'stable': bool(stable), 'hold_s': self.hold_time, 'success': self.success,
                'step': self.steps, 'seed': self.seed}

    def step(self, desired, valid=True):
        desired = np.asarray(desired, float)
        if desired.shape != (6,) or not np.isfinite(desired).all():
            raise ValueError('Task targets need six finite values')
        if not valid:
            self.target = np.clip(self.data.qpos[self.qadr] / self.ranges, 0, 1)
        else:
            increment = 2 * self.config.control_dt
            self.target += np.clip(np.clip(desired, 0, 1) - self.target, -increment, increment)
        self.data.ctrl[:] = self.target * self.ranges
        for _ in range(round(self.config.control_dt / self.model.opt.timestep)):
            self.mj.mj_step(self.model, self.data)
        self.steps += 1
        observation = self.observe()
        if observation['stable'] and self.hold_reference is None:
            self.hold_reference = observation['object_position'].copy()
        if not observation['stable']:
            self.hold_reference = None
        self.hold_time = self.hold_time + self.config.control_dt if observation['stable'] else 0.0
        self.success = self.success or self.hold_time >= self.config.hold_s
        observation.update(hold_s=self.hold_time, success=self.success)
        if not np.isfinite(observation['state']).all():
            raise RuntimeError('Nonfinite task physics state')
        return observation

    def rgb(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.config.height, width=self.config.width)
        self.renderer.update_scene(self.data, camera='task')
        return self.renderer.render().copy()

    def metadata(self):
        return {'task_version': TASK_VERSION, 'config': asdict(self.config), 'seed': self.seed,
                'initial_object_position': self.initial_position.tolist(), 'state_names': STATE_NAMES,
                'action_names': list(HAND_JOINTS), 'model_sha256': hashlib.sha256(self.xml.replace(str(DESCRIPTION), 'package://revo2_description').encode()).hexdigest(),
                'mujoco_version': self.mj.__version__, 'hardware': False,
                'render_backend': os.environ.get('MUJOCO_GL', 'platform default'),
                'collision_scope': 'primitive hand/object contacts; hand self-contact not validated',
                'success_definition': 'support removed, opposing finger contacts, bounded force/motion for configured hold time'}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
