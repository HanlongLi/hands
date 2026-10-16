"""Versioned human-to-Revo2 mapping datasets. Pure offline conversion; no hardware imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from r1hands.common.types import HAND_JOINTS
from r1hands.retarget.hand import HandRetargeter, hand_features
from r1hands.retarget.live import LiveHand, validate_landmarks

SCHEMA = 'r1hands.mapping.v1'
MODEL_SCHEMA = 'r1hands.mapping-model.v1'
MAPPER_VERSION = 'angles-one-euro-v1'
MAX_FRAMES = 3600


def mapper_from_profile(value):
    try:
        opened, closed = np.asarray(value['open'], float), np.asarray(value['closed'], float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Missing or invalid calibration') from error
    if (opened.shape != (6,) or closed.shape != (6,) or not np.isfinite(opened).all()
            or not np.isfinite(closed).all() or (np.abs(closed - opened) < 0.01).any()
            or (np.abs(opened) > 10).any() or (np.abs(closed) > 10).any()):
        raise ValueError('Invalid six-axis retargeting calibration')
    return HandRetargeter(open_feat=opened.copy(), fist_feat=closed.copy(), smooth=0)


def validate_clip(clip):
    if (not isinstance(clip, dict) or clip.get('schema') != 'r1hands.landmarks.v1'
            or clip.get('side') != 'right' or type(clip.get('swap')) is not bool):
        raise ValueError('Expected a right-hand landmark clip (version 1)')
    mapper_from_profile(clip.get('profile'))
    frames = clip.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError('Clip must contain 1–3600 frames')
    previous = -1
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError('Invalid frame object')
        t = frame.get('t')
        if not isinstance(t, (float, int)) or isinstance(t, bool) or not np.isfinite(t) or not previous < t <= 120000:
            raise ValueError('Frame times must increase within two minutes')
        previous = t
        if not isinstance(frame.get('hands'), list) or len(frame['hands']) > 2:
            raise ValueError('Expected zero to two hand observations')
        for hand in frame['hands']:
            if not isinstance(hand, dict):
                raise ValueError('Invalid hand object')
            if hand.get('label') not in ('left', 'right'):
                raise ValueError('Invalid detector handedness')
            points = np.asarray(hand.get('points'), float)
            image = hand.get('image')
            score = hand.get('score')
            if points.shape != (21, 3) or not np.isfinite(points).all() or (np.abs(points) > 10).any():
                raise ValueError('Expected 21 finite world landmarks')
            if (not isinstance(image, list) or len(image) != 21
                    or not all(isinstance(p, dict) and all(isinstance(p.get(k), (int, float))
                           and np.isfinite(p[k]) and abs(p[k]) <= 10 for k in ('x', 'y')) for p in image)):
                raise ValueError('Expected 21 finite image landmarks')
            if not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid handedness score')
    annotations = clip.get('annotations', [])
    if not isinstance(annotations, list) or len(annotations) > len(frames):
        raise ValueError('Invalid annotations')
    seen = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError('Invalid annotation object')
        index = annotation.get('frame_index')
        target = np.asarray(annotation.get('position'), float)
        if (type(index) is not int or not 0 <= index < len(frames) or index in seen
                or target.shape != (6,) or not np.isfinite(target).all()
                or (target < 0).any() or (target > 1).any()):
            raise ValueError('Annotations need unique frame indices and six targets in [0,1]')
        seen.add(index)
        for key in ('note', 'reviewer'):
            if key in annotation and (not isinstance(annotation[key], str) or len(annotation[key]) > 500):
                raise ValueError(f'Invalid annotation {key}')
    metadata = clip.get('metadata', {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) or len(v) > 200 for v in metadata.values()):
        raise ValueError('Clip metadata must contain short text values')
    return clip


def select_right(frame, swap):
    """Match the browser's identity, clipping and duplicate-label gates."""
    candidates, anatomical = [], []
    for hand in frame['hands']:
        # This is the same explicit raw-image handedness convention used in the panel.
        side = 'right' if hand['label'] == 'left' else 'left'
        if swap:
            side = 'left' if side == 'right' else 'right'
        if side != 'right':
            continue
        anatomical.append(hand)
        clipped = any(p['x'] < 0.01 or p['x'] > 0.99 or p['y'] < 0.01 or p['y'] > 0.99 for p in hand['image'])
        if hand['score'] >= 0.8 and not clipped:
            candidates.append(hand)
    if len(candidates) > 1:
        return None, None, 'ambiguous_identity'
    if len(candidates) == 1:
        return candidates[0], candidates[0], None
    if not anatomical:
        return None, None, 'missing_right_hand'
    raw = max(anatomical, key=lambda h: h['score'])
    return raw, None, 'low_identity_score' if raw['score'] < 0.8 else 'clipped_image'


def palm_coordinates(points):
    """Wrist-centered, palm-aligned, scale-normalized landmarks; axes in metadata."""
    kp = validate_landmarks(points)
    forward = kp[9] - kp[0]
    scale = np.linalg.norm(forward)
    forward /= scale
    lateral = kp[5] - kp[17]
    lateral -= np.dot(lateral, forward) * forward
    norm = np.linalg.norm(lateral)
    if norm < scale * 1e-5:
        raise ValueError('Degenerate palm frame')
    lateral /= norm
    normal = np.cross(lateral, forward)
    return (kp - kp[0]) @ np.column_stack((lateral, forward, normal)) / scale


def convert_clip(clip):
    validate_clip(clip)
    mapper = mapper_from_profile(clip['profile'])
    live = LiveHand(mapper)
    n = len(clip['frames'])
    data = {
        'time_s': np.array([f['t'] / 1000 for f in clip['frames']], np.float64),
        'inference_ms': np.array([f.get('inference_ms') if isinstance(f.get('inference_ms'), (int, float)) else np.nan for f in clip['frames']], np.float32),
        'human_landmarks_world': np.full((n, 21, 3), np.nan, np.float32),
        'human_landmarks_image': np.full((n, 21, 2), np.nan, np.float32),
        'human_landmarks_palm': np.full((n, 21, 3), np.nan, np.float32),
        'human_features': np.full((n, 6), np.nan, np.float32),
        'raw_robot_target': np.full((n, 6), np.nan, np.float32),
        'filtered_robot_target': np.zeros((n, 6), np.float32),
        'corrected_robot_target': np.full((n, 6), np.nan, np.float32),
        'handedness_score': np.full(n, np.nan, np.float32),
        'observation_valid': np.zeros(n, bool), 'tracking_valid': np.zeros(n, bool),
        'label_valid': np.zeros(n, bool), 'training_mask': np.zeros(n, bool),
        'collision_checked': np.zeros(n, bool),
        'quality_reason': np.full(n, '', dtype='U32'),
        'label_source': np.full(n, 'unavailable', dtype='U16'),
        'reviewer': np.full(n, '', dtype='U500'), 'label_note': np.full(n, '', dtype='U500'),
    }
    previous = np.zeros(6)
    for i, frame in enumerate(clip['frames']):
        raw, accepted, reason = select_right(frame, clip['swap'])
        if raw is not None:
            data['human_landmarks_world'][i] = raw['points']
            data['human_landmarks_image'][i] = [[p['x'], p['y']] for p in raw['image']]
            data['handedness_score'][i] = raw['score']
        if accepted is None:
            live.reset()
        else:
            try:
                kp = validate_landmarks(accepted['points'])
                canonical = palm_coordinates(kp)
                data['human_landmarks_palm'][i] = canonical
                data['human_features'][i] = hand_features(kp)
                data['observation_valid'][i] = True
                data['raw_robot_target'][i] = mapper(kp)
                target = live.update(kp, accepted['score'], data['time_s'][i])
                if target is not None:
                    previous = target
                    data['tracking_valid'][i] = True
                    reason = 'valid'
                else:
                    reason = 'acquiring'
            except ValueError:
                live.reset()
                reason = 'invalid_geometry'
        data['filtered_robot_target'][i] = previous
        data['quality_reason'][i] = reason
    for label in clip.get('annotations', []):
        i = label['frame_index']
        data['corrected_robot_target'][i] = label['position']
        data['label_valid'][i] = True
        data['label_source'][i] = 'operator_review'
        data['reviewer'][i] = label.get('reviewer', '')
        data['label_note'][i] = label.get('note', '')
    data['training_mask'] = data['observation_valid'] & data['label_valid']
    # Group by the observations themselves, independent of profile or edited labels.
    # Re-exporting the same clip must not move its samples into a different split.
    identity = json.dumps({'frames': [{'t': f['t'], 'hands': f['hands']} for f in clip['frames']], 'swap': clip['swap']}, sort_keys=True, separators=(',', ':'), allow_nan=False)
    group = hashlib.sha256(identity.encode()).hexdigest()
    reasons, counts = np.unique(data['quality_reason'], return_counts=True)
    metadata = {
        'schema': SCHEMA, 'mapper_version': MAPPER_VERSION, 'clip_group': group,
        'source_schema': clip['schema'], 'created_at': clip.get('created_at'), 'swap': clip['swap'],
        'human_joint_order': 'MediaPipe 21 landmarks: wrist, thumb, index, middle, ring, pinky',
        'clip_metadata': clip.get('metadata', {}), 'synthetic': bool(clip.get('synthetic', False)),
        'profile': clip['profile'], 'joint_names': list(HAND_JOINTS),
        'units': {'world': 'MediaPipe estimated meters', 'image': 'normalized image coordinates',
                  'palm': 'palm-length units', 'human_features': 'radians', 'robot_target': 'normalized [0,1]', 'time': 'seconds'},
        'palm_axes': ['index-minus-pinky, orthogonalized', 'wrist-to-middle-MCP', 'lateral cross forward'],
        'provenance': {'raw_robot_target': 'analytic mapper pseudo-label',
                       'filtered_robot_target': 'analytic mapper with acquisition and One Euro smoothing',
                       'corrected_robot_target': 'operator-provided pose, NaN when unavailable',
                       'collision_checked': 'false: no collision validator implemented'},
        'filter': {'min_cutoff': 1.5, 'beta': 0.7, 'min_identity_score': 0.8, 'acquire_frames': 3},
        'summary': {'frames': n, 'valid_observations': int(data['observation_valid'].sum()),
                    'reviewed_labels': int(data['label_valid'].sum()), 'training_samples': int(data['training_mask'].sum()),
                    'quality_counts': dict(zip(reasons.tolist(), counts.tolist()))},
    }
    data['metadata_json'] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    return data


def export_clip(clip, destination):
    data = convert_clip(clip)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **data)
    return json.loads(str(data['metadata_json']))


@dataclass
class MappingModel:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict

    def predict_features(self, features):
        x = np.asarray(features, float)
        if x.ndim not in (1, 2) or x.shape[-1] != 6 or not np.isfinite(x).all():
            raise ValueError('Expected six finite human joint features')
        scalar = x.ndim == 1
        x = np.atleast_2d(x)
        design = np.column_stack(((x - self.mean) / self.scale, np.ones(len(x))))
        prediction = np.clip(design @ self.weights, 0, 1)
        return prediction[0] if scalar else prediction


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        weights, mean, scale = (np.array(data[k], float) for k in ('weights', 'feature_mean', 'feature_scale'))
    if (metadata.get('schema') != MODEL_SCHEMA or weights.shape != (7, 6)
            or mean.shape != (6,) or scale.shape != (6,) or (scale <= 0).any()
            or not all(np.isfinite(v).all() for v in (weights, mean, scale))):
        raise ValueError('Invalid mapping model')
    return MappingModel(weights, mean, scale, metadata)


class LearnedRetargeter:
    """The learned baseline can drive the read-only preview, never the HAL."""
    smooth = 0

    def __init__(self, model):
        self.model = model

    def __call__(self, landmarks):
        return self.model.predict_features(hand_features(landmarks))


def _samples(paths, target):
    rows, seen_clips = [], set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json']))
            if metadata.get('schema') != SCHEMA:
                raise ValueError(f'{path}: unsupported mapping dataset')
            mask = data['training_mask'] if target == 'reviewed' else data['observation_valid']
            y = data['corrected_robot_target'] if target == 'reviewed' else data['raw_robot_target']
            group = metadata['clip_group']
            if group in seen_clips:
                raise ValueError('The same source clip was supplied more than once')
            seen_clips.add(group)
            operator = metadata.get('clip_metadata', {}).get('operator_id', '').strip()
            for i in np.flatnonzero(mask):
                rows.append((data['human_features'][i].astype(float), y[i].astype(float),
                             data['raw_robot_target'][i].astype(float), group, operator, metadata.get('synthetic', False), str(path)))
    if not rows:
        raise ValueError('No usable labels. Add reviewed poses in the panel, or explicitly choose --target teacher.')
    if not all(np.isfinite(r[0]).all() and np.isfinite(r[1]).all() for r in rows):
        raise ValueError('Nonfinite training inputs or labels')
    # Multiple exports of the same clip cannot silently weight one recording several times.
    return rows


def _metrics(prediction, target, baseline):
    return {'mae': float(np.abs(prediction - target).mean()),
            'per_joint_mae': dict(zip(HAND_JOINTS, np.abs(prediction - target).mean(0).tolist())),
            'analytic_baseline_mae': float(np.abs(baseline - target).mean()), 'samples': len(target)}


def train(paths, destination, target='reviewed', seed=0, ridge=0.01, split_by='clip'):
    if target not in ('reviewed', 'teacher') or split_by not in ('clip', 'operator'):
        raise ValueError('Unknown target or split mode')
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Ridge regularization must be positive')
    rows = _samples(paths, target)
    operators = {r[3]: r[4] for r in rows}
    if split_by == 'operator' and not all(operators.values()):
        raise ValueError('Operator split requires operator_id on every clip')
    groups = np.array([r[4] if split_by == 'operator' else r[3] for r in rows])
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('At least two distinct labeled clips/operators are required for held-out evaluation')
    validation_groups = np.random.default_rng(seed).permutation(unique)[:max(1, int(np.ceil(len(unique) * 0.2)))]
    validation = np.isin(groups, validation_groups)
    x, y, baseline = (np.stack([r[i] for r in rows]) for i in range(3))
    mean, scale = x[~validation].mean(0), x[~validation].std(0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    reg = np.eye(design.shape[1]) * ridge
    reg[-1, -1] = 0
    a = design[~validation]
    weights = np.linalg.solve(a.T @ a + reg, a.T @ y[~validation])
    prediction = np.clip(design @ weights, 0, 1)
    report = {
        'schema': MODEL_SCHEMA, 'model': 'ridge-regression on six human flexion/opposition features',
        'target_source': target, 'split_by': split_by, 'seed': seed, 'ridge': ridge,
        'train_groups': sorted(set(groups[~validation].tolist())), 'validation_groups': validation_groups.tolist(),
        'train_datasets': sorted({r[6] for i, r in enumerate(rows) if not validation[i]}),
        'validation_datasets': sorted({r[6] for i, r in enumerate(rows) if validation[i]}),
        'train_clip_groups': sorted({r[3] for i, r in enumerate(rows) if not validation[i]}),
        'validation_clip_groups': sorted({r[3] for i, r in enumerate(rows) if validation[i]}),
        'train': _metrics(prediction[~validation], y[~validation], baseline[~validation]),
        'validation': _metrics(prediction[validation], y[validation], baseline[validation]),
        'label_interpretation': 'operator-reviewed kinematic poses' if target == 'reviewed' else 'distillation of existing analytic mapper; not independent ground truth',
        'synthetic_samples': sum(bool(r[5]) for r in rows),
        'collision_validated': False, 'hardware_validated': False,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, weights=weights, feature_mean=mean, feature_scale=scale,
                        metadata_json=np.array(json.dumps(report, sort_keys=True)))
    return report


def evaluate(model_path, paths):
    with np.load(model_path, allow_pickle=False) as model:
        report = json.loads(str(model['metadata_json']))
        if report.get('schema') != MODEL_SCHEMA:
            raise ValueError('Unsupported mapping model')
        rows = _samples(paths, report['target_source'])
        train_groups = set(report['train_clip_groups'])
        if any(r[3] in train_groups for r in rows):
            raise ValueError('Evaluation includes a training clip; provide held-out clips only')
        if report['split_by'] == 'operator' and (not all(r[4] for r in rows) or any(r[4] in set(report['train_groups']) for r in rows)):
            raise ValueError('Operator evaluation requires known, held-out operators')
        x, y, baseline = (np.stack([r[i] for r in rows]) for i in range(3))
        design = np.column_stack(((x - model['feature_mean']) / model['feature_scale'], np.ones(len(x))))
        return _metrics(np.clip(design @ model['weights'], 0, 1), y, baseline)


def main():
    parser = argparse.ArgumentParser(description='Offline human-to-Revo2 mapping data and baseline training')
    sub = parser.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('export')
    exp.add_argument('clips', nargs='+', type=Path)
    exp.add_argument('--output', required=True, type=Path)
    fit = sub.add_parser('train')
    fit.add_argument('datasets', nargs='+', type=Path)
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--target', choices=['reviewed', 'teacher'], default='reviewed')
    fit.add_argument('--split-by', choices=['clip', 'operator'], default='clip')
    fit.add_argument('--seed', type=int, default=0)
    fit.add_argument('--ridge', type=float, default=0.01)
    score = sub.add_parser('evaluate')
    score.add_argument('model', type=Path)
    score.add_argument('datasets', nargs='+', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        if len({p.stem for p in args.clips}) != len(args.clips):
            parser.error('Clip filenames must have unique stems in one export')
        args.output.mkdir(parents=True, exist_ok=True)
        for path in args.clips:
            if path.stat().st_size > 20 * 1024 * 1024:
                parser.error(f'{path}: clip exceeds 20 MiB')
            metadata = export_clip(json.loads(path.read_text()), args.output / f'{path.stem}.npz')
            (args.output / f'{path.stem}.manifest.json').write_text(json.dumps(metadata, indent=2))
            print(json.dumps({'clip': str(path), **metadata['summary']}))
    elif args.command == 'train':
        report = train(args.datasets, args.output, args.target, args.seed, args.ridge, args.split_by)
        args.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(evaluate(args.model, args.datasets), indent=2))


if __name__ == '__main__':
    main()
