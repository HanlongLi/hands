"""Read-only hand visualization and retargeting; this module has no HAL dependency."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from r1hands.common.robot_model import REPO
from r1hands.common.types import HAND_JOINTS
from r1hands.retarget.hand import HandRetargeter
from r1hands.retarget.live import LiveHand
from r1hands.data.mapping import mapper_from_profile, LearnedRetargeter
from r1hands.sim.build_model import HAND_DRIVE, HAND_RANGE

DESCRIPTION = REPO / 'assets' / 'revo2_description'
MESH_DIR = DESCRIPTION / 'meshes' / 'revo2_right_hand'


def profile(retarget):
    return {'open': retarget.open_feat.tolist(), 'closed': retarget.fist_feat.tolist()}


def validate_profile(value):
    return mapper_from_profile(value)


class HandPreview:
    """Replay advances the filter on recorded sample time, independent of playback speed."""
    def __init__(self, model=None):
        self.model = model
        self.stream = None
        self.last_seq = -1
        self.last_sample = None
        self.signature = None
        self.live = LiveHand(LearnedRetargeter(model) if model else HandRetargeter(smooth=0))
        self.target = np.zeros(6)
        self.source = 'idle'
        self.calibration = 'current'
        self.last_received = 0.0

    def update(self, message, current_profile):
        seq = int(message['seq'])
        sample = float(message['sample_time'])
        if not np.isfinite(sample) or sample < 0:
            raise ValueError('Invalid preview sample timestamp')
        source = message.get('source', 'live')
        if source not in ('live', 'replay'):
            raise ValueError('Unknown preview source')
        calibration = message.get('profile') if source == 'replay' else current_profile
        calibration = calibration or current_profile
        mapper = validate_profile(calibration)
        signature = (tuple(mapper.open_feat), tuple(mapper.fist_feat))
        stream = str(message['stream'])[:100]
        if stream != self.stream or signature != self.signature:
            self.live = LiveHand(LearnedRetargeter(self.model) if self.model else mapper)
            if source == 'replay':
                self.target = np.zeros(6)
            self.stream, self.signature = stream, signature
            self.last_seq, self.last_sample = -1, None
        if seq <= self.last_seq or (self.last_sample is not None and sample <= self.last_sample):
            return False
        self.last_seq, self.last_sample = seq, sample
        self.last_received = time.monotonic()
        self.source = source
        self.calibration = 'recorded' if source == 'replay' and message.get('profile') else 'current'
        observation = message.get('hand')
        if observation is None:
            self.live.reset()
            return True
        result = self.live.update(observation.get('points'), float(observation.get('score', 0)), sample)
        if result is not None:
            self.target = result
        return True

    def snapshot(self):
        status = self.live.status
        if self.source == 'live' and time.monotonic() - self.last_received > 0.25:
            status = 'held'
        return {'target': self.target.tolist(), 'status': status,
                'source': self.source, 'seq': self.last_seq, 'calibration': self.calibration, 'stream': self.stream,
                'mapper': 'learned baseline (preview only)' if self.model else 'analytic v1'}


def _origin(element):
    origin = element.find('origin')
    return {name: [float(v) for v in origin.get(name, '0 0 0').split()] if origin is not None else [0, 0, 0]
            for name in ('xyz', 'rpy')}


@lru_cache
def hand_model():
    root = ET.parse(DESCRIPTION / 'urdf' / 'revo2_right_hand.urdf').getroot()
    links = []
    for link in root.findall('link'):
        visuals = []
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is not None:
                filename = Path(mesh.get('filename')).name
                visuals.append({**_origin(visual), 'mesh': f'/preview/meshes/{filename}',
                                'scale': [float(v) for v in mesh.get('scale', '1 1 1').split()]})
        links.append({'name': link.get('name'), 'visuals': visuals})
    joints = []
    for joint in root.findall('joint'):
        mimic = joint.find('mimic')
        axis = joint.find('axis')
        joints.append({'name': joint.get('name'), 'type': joint.get('type'), **_origin(joint),
                       'parent': joint.find('parent').get('link'), 'child': joint.find('child').get('link'),
                       'axis': [float(v) for v in axis.get('xyz').split()] if axis is not None else [0, 0, 1],
                       'mimic': dict(mimic.attrib) if mimic is not None else None})
    return {'root': 'right_base_link', 'links': links, 'joints': joints,
            'drives': [{'joint': f'right_{HAND_DRIVE[j]}', 'range': HAND_RANGE[j]} for j in HAND_JOINTS]}


def example_clip():
    """Synthetic open/close observations for testing the viewer, explicitly labeled."""
    frames = []
    for i in range(180):
        curl = 0.05 + 0.95 * (0.5 - 0.5 * np.cos(2 * np.pi * i / 90))
        kp = np.zeros((21, 3))
        for start, y in ((5, 0.02), (9, 0), (13, -0.02), (17, -0.04)):
            point = np.array([0.09, y, 0.0])
            kp[start] = point
            for j in range(1, 4):
                point = point + 0.025 * np.array([np.cos(curl * j), 0, -np.sin(curl * j)])
                kp[start + j] = point
        kp[1:5] = [[0.03, 0.03, 0], [0.05, 0.05, 0], [0.07, 0.06, 0], [0.09, 0.065, 0]]
        image = [{'x': float(0.5 + p[1] * 3), 'y': float(0.9 - p[0] * 4), 'z': float(p[2])} for p in kp]
        frames.append({'t': i * 1000 / 30, 'hands': [{'points': kp.tolist(), 'image': image,
                                                     'label': 'left', 'score': 0.99}], 'inference_ms': 0})
    return {'schema': 'r1hands.landmarks.v1', 'side': 'right', 'synthetic': True,
            'created_at': None, 'swap': False, 'profile': profile(HandRetargeter()), 'frames': frames}
