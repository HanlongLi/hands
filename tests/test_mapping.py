import copy
import io
import json
import numpy as np
import pytest

from r1hands.data.mapping import (convert_clip, export_clip, train, evaluate, load_model,
                                  palm_coordinates, validate_clip, select_right)
from r1hands.panel.preview import example_clip, HandPreview
from tests.test_core import _synthetic_hand
from tests.test_panel import _panel


def reviewed_clip(offset=0, operator='operator-a'):
    clip = example_clip()
    clip['frames'] = clip['frames'][:30]
    clip['metadata'] = {'operator_id': operator, 'session_id': str(offset)}
    for i, frame in enumerate(clip['frames']):
        frame['hands'][0]['points'] = (_synthetic_hand(0.1 + i * 0.025 + offset)).tolist()
    data = convert_clip(clip)
    clip['annotations'] = [{'frame_index': i, 'position': np.clip(data['raw_robot_target'][i] * 0.8 + 0.05, 0, 1).tolist(),
                            'reviewer': 'tester', 'note': 'reviewed pose'} for i in range(0, 30, 3)]
    return clip


def test_mapping_export_has_aligned_labels_masks_and_provenance():
    clip = reviewed_clip()
    clip['frames'][3]['hands'] = []
    clip['frames'][7]['hands'][0]['score'] = 0.2
    clip['frames'][9]['hands'][0]['image'][0]['x'] = 0
    clip['frames'][11]['hands'] *= 2
    before = copy.deepcopy(clip)
    data = convert_clip(clip)
    assert clip == before
    assert data['human_landmarks_world'].shape == (30, 21, 3)
    assert data['raw_robot_target'].shape == (30, 6)
    assert data['quality_reason'][3] == 'missing_right_hand'
    assert data['quality_reason'][7] == 'low_identity_score'
    assert data['quality_reason'][9] == 'clipped_image'
    assert data['quality_reason'][11] == 'ambiguous_identity'
    assert not data['observation_valid'][3]
    assert data['label_valid'][3] and not data['training_mask'][3]
    assert np.isnan(data['corrected_robot_target'][1]).all()
    assert not data['collision_checked'].any()
    assert np.isfinite(data['human_landmarks_world'][7]).all()
    metadata = json.loads(str(data['metadata_json']))
    assert metadata['schema'] == 'r1hands.mapping.v1'
    assert metadata['summary']['reviewed_labels'] == 10
    assert 'pseudo-label' in metadata['provenance']['raw_robot_target']


def test_export_filtered_targets_match_preview_replay():
    clip = reviewed_clip()
    clip['frames'][10]['hands'] = []
    data = convert_clip(clip)
    preview = HandPreview()
    for i, frame in enumerate(clip['frames']):
        _, accepted, _ = select_right(frame, clip['swap'])
        preview.update({'seq': i, 'sample_time': frame['t']/1000, 'stream':'test',
                        'source':'replay', 'profile':clip['profile'], 'hand': accepted}, clip['profile'])
        np.testing.assert_allclose(data['filtered_robot_target'][i], preview.target, atol=1e-6)


def test_palm_coordinates_are_invariant_to_rigid_transforms_and_scale():
    kp = _synthetic_hand(0.5)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(palm_coordinates(kp), palm_coordinates(kp @ rotation.T * 2.4 + [2,3,4]), atol=1e-8)


def test_invalid_clips_and_annotations_are_rejected():
    clip = reviewed_clip()
    clip['annotations'].append(copy.deepcopy(clip['annotations'][0]))
    with pytest.raises(ValueError, match='unique'):
        validate_clip(clip)
    clip = reviewed_clip()
    clip['frames'][1]['t'] = clip['frames'][0]['t']
    with pytest.raises(ValueError, match='times'):
        validate_clip(clip)
    clip = reviewed_clip()
    clip['annotations'][0]['position'][0] = 1.1
    with pytest.raises(ValueError):
        validate_clip(clip)


def test_clip_identity_ignores_review_edits_and_inference_timing():
    a = reviewed_clip()
    b = copy.deepcopy(a)
    b['annotations'] = []
    b['profile']['open'][0] += 0.01
    b['frames'][0]['inference_ms'] = 123
    a_meta = json.loads(str(convert_clip(a)['metadata_json']))
    b_meta = json.loads(str(convert_clip(b)['metadata_json']))
    assert a_meta['clip_group'] == b_meta['clip_group']


def test_train_grouped_baseline_and_reject_training_clip_evaluation(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f'clip-{i}.npz'
        export_clip(reviewed_clip(i * 0.02, f'operator-{i}'), path)
        paths.append(path)
    model_path = tmp_path / 'model.npz'
    report = train(paths, model_path, split_by='operator')
    assert not set(report['train_groups']) & set(report['validation_groups'])
    assert report['validation']['mae'] < 0.05
    assert report['target_source'] == 'reviewed'
    assert not report['collision_validated'] and not report['hardware_validated']
    model = load_model(model_path)
    assert model.predict_features(np.zeros(6)).shape == (6,)
    assert np.isfinite(model.weights).all()
    with pytest.raises(ValueError, match='training clip'):
        evaluate(model_path, paths)
    held_out = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data['metadata_json']))['clip_group'] in report['validation_clip_groups']:
                held_out.append(path)
    assert evaluate(model_path, held_out)['samples'] == report['validation']['samples']
    with pytest.raises(ValueError, match='same source clip'):
        train([paths[0], paths[0]], tmp_path / 'duplicate.npz')
    with pytest.raises(ValueError, match='At least two'):
        train([paths[0]], tmp_path / 'single.npz')


def test_default_training_requires_reviewed_labels_and_teacher_is_explicit(tmp_path):
    paths = []
    for i in range(2):
        clip = reviewed_clip(i * 0.03)
        clip['annotations'] = []
        path = tmp_path / f'raw-{i}.npz'
        export_clip(clip, path)
        paths.append(path)
    with pytest.raises(ValueError, match='No usable labels'):
        train(paths, tmp_path/'model.npz')
    report = train(paths, tmp_path/'teacher.npz', target='teacher')
    assert 'not independent ground truth' in report['label_interpretation']


def test_learned_model_only_changes_preview(tmp_path, monkeypatch):
    paths = []
    for i in range(2):
        path = tmp_path / f'clip-{i}.npz'
        export_clip(reviewed_clip(i*0.03), path)
        paths.append(path)
    model_path = tmp_path/'model.npz'
    train(paths, model_path)
    panel = _panel(tmp_path, monkeypatch)
    panel.preview = HandPreview(load_model(model_path))
    before = panel.runner.target['right'].copy()
    try:
        for i in range(4):
            panel.handle({'type':'preview', 'seq':i, 'sample_time':i/30, 'stream':'learned', 'source':'replay',
                          'hand':{'points':_synthetic_hand(0.7).tolist(), 'score':0.99}})
        assert panel.preview.snapshot()['mapper'].startswith('learned')
        assert panel.preview.target[2:].mean() > 0.1
        np.testing.assert_equal(before, panel.runner.target['right'])
    finally:
        panel.runner.close()
        panel.robot.close()


def test_mapping_export_http_is_read_only_and_reports_bad_input(tmp_path, monkeypatch):
    import asyncio
    from aiohttp.test_utils import TestClient, TestServer
    from r1hands.panel.app import make_app

    async def run():
        panel = _panel(tmp_path, monkeypatch)
        client = TestClient(TestServer(make_app(panel)))
        before = panel.runner.target['right'].copy()
        try:
            await client.start_server()
            response = await client.post('/mapping/export', json=reviewed_clip())
            assert response.status == 200
            with np.load(io.BytesIO(await response.read()), allow_pickle=False) as dataset:
                assert dataset['training_mask'].sum() == 10
                assert 'metadata_json' in dataset.files
            np.testing.assert_equal(before, panel.runner.target['right'])
            response = await client.post('/mapping/export', json={'schema':'bad'})
            assert response.status == 400
        finally:
            await client.close()
            panel.runner.close()
            panel.robot.close()
    asyncio.run(run())
