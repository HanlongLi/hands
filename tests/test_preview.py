import numpy as np
import pytest

from r1hands.panel.preview import HandPreview, MESH_DIR, example_clip, hand_model, profile, validate_profile
from r1hands.retarget.hand import HandRetargeter
from tests.test_core import _synthetic_hand
from tests.test_panel import _panel


def message(index, *, stream='clip-a', sample=None, kp=None):
    return {'type': 'preview', 'source': 'replay', 'stream': stream, 'seq': index,
            'sample_time': index / 30 if sample is None else sample,
            'hand': {'points': (kp if kp is not None else _synthetic_hand(1)).tolist(), 'score': 0.99},
            'profile': profile(HandRetargeter())}


def test_preview_replay_is_read_only_even_on_a_panel(tmp_path, monkeypatch):
    panel = _panel(tmp_path, monkeypatch)
    try:
        before = panel.runner.target['right'].copy()
        for i in range(30):
            panel.handle(message(i))
        assert panel.preview.snapshot()['source'] == 'replay'
        assert panel.preview.target[2:].min() > 0.8
        np.testing.assert_equal(panel.runner.target['right'], before)
        assert not panel.teleop_armed
        panel.teleop_armed = True
        with pytest.raises(ValueError, match='Disarm'):
            panel.handle(message(31))
    finally:
        panel.runner.close()
        panel.robot.close()


def test_replaying_the_same_samples_produces_the_same_targets():
    mapper = HandPreview()
    outputs = []
    for stream in ('first-run', 'second-run'):
        run = []
        for i in range(30):
            mapper.update(message(i, stream=stream, kp=_synthetic_hand(0.05 + i / 35)), profile(HandRetargeter()))
            run.append(mapper.target.copy())
        outputs.append(run)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)


def test_preview_rejects_reordered_samples_and_reacquires_after_gaps():
    mapper = HandPreview()
    current = profile(HandRetargeter())
    for i in range(3):
        mapper.update(message(i), current)
    previous = mapper.target.copy()
    mapper.update(message(1, kp=_synthetic_hand(0.05)), current)
    np.testing.assert_equal(mapper.target, previous)
    mapper.update({**message(3), 'hand': None}, current)
    assert mapper.live.status == 'waiting'
    mapper.update(message(4), current)
    assert mapper.live.status == 'acquiring'
    with pytest.raises(ValueError):
        mapper.update(message(5, sample=float('nan')), current)
    with pytest.raises(ValueError):
        validate_profile({'open': [0]*6, 'closed': [0]*6})


def test_model_matches_revo2_assets_and_mimic_joints():
    model = hand_model()
    assert len(model['drives']) == 6
    assert len(model['links']) == 22
    assert sum(j['mimic'] is not None for j in model['joints']) == 5
    for link in model['links']:
        for visual in link['visuals']:
            assert (MESH_DIR / visual['mesh'].rsplit('/', 1)[1]).is_file()
    example = example_clip()
    assert example['synthetic'] and len(example['frames']) == 180
    validate_profile(example['profile'])


def test_preview_model_and_mesh_http_routes(tmp_path, monkeypatch):
    import asyncio
    from aiohttp.test_utils import TestClient, TestServer
    from r1hands.panel.app import make_app

    async def run():
        panel = _panel(tmp_path, monkeypatch)
        client = TestClient(TestServer(make_app(panel)))
        try:
            await client.start_server()
            response = await client.get('/preview/model')
            assert response.status == 200
            model = await response.json()
            mesh = model['links'][0]['visuals'][0]['mesh']
            response = await client.get(mesh)
            assert response.status == 200 and len(await response.read()) > 84
            response = await client.get('/preview/meshes/LICENSE')
            assert response.status == 404
            response = await client.get('/preview/example')
            example = await response.json()
            assert example['schema'] == 'r1hands.landmarks.v1'
            assert example['synthetic']
        finally:
            await client.close()
            panel.runner.close()
            panel.robot.close()
    asyncio.run(run())
