import json
from pathlib import Path
import time
import numpy as np
import pytest

mujoco=pytest.importorskip('mujoco')
from r1hands.sim.hand_task import HandTask, TaskConfig
from r1hands.tasks.episodes import EpisodeRecorder
from r1hands.tasks.policy import train, evaluate, load_policy
from r1hands.tasks.service import TaskService

GRASP=np.array([.734,.382,.323,.541,.78,.626])


def test_task_reset_reproducibility_and_no_supported_false_success():
    env=HandTask()
    try:
        a=env.reset(7)['state'].copy()
        for _ in range(10):obs=env.step(np.zeros(6))
        assert not obs['success'] and obs['contacts']['external_support']
        np.testing.assert_array_equal(a,env.reset(7)['state'])
        assert not np.array_equal(a,env.reset(8)['state'])
        env.release_support()
        for _ in range(40):obs=env.step(np.zeros(6))
        assert not obs['success'] and obs['object_position'][2]<0.05
        with pytest.raises(ValueError):env.step([float('nan')]*6)
    finally:env.close()


def test_opposing_grasp_succeeds_only_after_support_withdrawal():
    env=HandTask()
    try:
        for _ in range(25):obs=env.step(GRASP)
        assert not obs['success']
        env.release_support()
        for _ in range(40):obs=env.step(GRASP)
        assert obs['success'] and obs['hold_s']>=1
        assert obs['contacts']['opposing'] and not obs['contacts']['external_support']
        assert obs['contacts']['max_force_n']<env.config.force_limit_n
    finally:env.close()


def collect(directory,seed,grasp=True):
    env=HandTask(TaskConfig(width=64,height=48,max_steps=100))
    try:
        env.reset(seed)
        recorder=EpisodeRecorder(env,directory,{'operator_id':'automated-test','source':'scripted fixture'})
        q=GRASP if grasp else np.zeros(6)
        for step in range(65):
            if step==25:
                env.release_support();recorder.release_event(env)
            env.step(q)
            recorder.append(env,q,'scripted',True,0)
        meta=recorder.finish(env)
        return Path(directory)/(meta['episode_id']+'.npz'),meta
    finally:env.close()


def test_episode_alignment_rgb_and_provenance(tmp_path):
    path,metadata=collect(tmp_path,0)
    with np.load(path,allow_pickle=False) as data:
        assert data['observation_state'].shape==(66,26)
        assert data['action'].shape==(65,6)
        assert data['observation_rgb'].shape==(66,48,64,3)
        assert data['observation_rgb'].dtype==np.uint8
        np.testing.assert_allclose(np.diff(data['time_s']),0.05,atol=1e-8)
        assert data['observation_state'][25,25]==1 # release event observed before next action
        assert metadata['events'][0]['observation_index']==25
        assert metadata['success']
        assert metadata['source_counts']=={'scripted':65}
        assert not metadata['hardware']
        assert np.isfinite(data['action']).all()
        assert np.max(np.abs(np.diff(data['action'],axis=0)))<=0.10001
        assert (data['observation_rgb'][0]!=data['observation_rgb'][-1]).any()
    assert path.with_suffix('.json').is_file()


def test_task_policy_held_out_seeds_and_closed_loop_evaluation(tmp_path):
    paths=[collect(tmp_path,i)[0] for i in range(2)]
    model=tmp_path/'policy.npz'
    report=train(paths,model)
    assert not set(report['train_seeds'])&set(report['validation_seeds'])
    assert report['source_counts']=={'scripted':130}
    predict,metadata=load_policy(model)
    with np.load(paths[0],allow_pickle=False) as data:
        assert predict(data['observation_state'][0]).shape==(6,)
    result=evaluate(model,report['validation_seeds'])
    assert len(result['rollouts'])==1 and result['simulation_only']
    with pytest.raises(ValueError,match='held-out'):evaluate(model,report['train_seeds'])
    with pytest.raises(ValueError,match='different reset seeds'):train([paths[0]],tmp_path/'bad.npz')
    with pytest.raises(ValueError,match='Duplicate'):train([paths[0],paths[0]],tmp_path/'duplicate.npz')
    failure,_=collect(tmp_path,9,grasp=False)
    with pytest.raises(ValueError,match='No usable successful'):train([failure],tmp_path/'failure.npz')


def test_task_process_watchdog_recording_and_download_files(tmp_path):
    service=TaskService(tmp_path)
    try:
        assert not service.snapshot()['ready']
        service.start();service.call('reset',seed=0);service.call('begin',provenance={'operator_id':'test'})
        with pytest.raises(ValueError,match='Finish'):service.call('reset',seed=1)
        service.set_target(GRASP,'mapping',valid=False)
        time.sleep(0.2)
        saved=service.call('finish')['saved']
        with np.load(tmp_path/(saved['episode_id']+'.npz'),allow_pickle=False) as data:
            assert len(data['action'])>=1
            assert (~data['input_valid']).any()
        service.hold(lock=True)
        service.set_target(np.ones(6))
        assert service.source=='hold'
    finally:service.close()
    assert not service.thread.is_alive()


def test_task_mapping_ignores_replays_stale_frames_and_robot_arming(tmp_path, monkeypatch):
    from tests.test_panel import _panel
    from tests.test_preview import message
    panel=_panel(tmp_path,monkeypatch)
    try:
        panel.task_service.mapping_enabled=True
        initial_t=panel.task_service.target_t
        panel.handle(message(0)) # replay cannot become a task command
        assert panel.task_service.target_t==initial_t
        for seq in range(3):panel.handle({**message(seq,stream='camera'),'source':'live'})
        assert panel.task_service.source=='mapping'
        assert panel.task_service.target_t>initial_t
        accepted_t=panel.task_service.target_t
        panel.handle({**message(1,stream='camera'),'source':'live'})
        assert panel.task_service.target_t==accepted_t # duplicates cannot refresh watchdog
        with pytest.raises(ValueError,match='Stop task mapping'):panel.handle({'type':'teleop_arm','on':True})
        panel.handle({'type':'estop'})
        assert panel.task_service.locked and not panel.task_service.mapping_enabled
        panel.handle({'type':'resume'})
        assert not panel.task_service.locked and panel.task_service.source=='hold'
    finally:
        panel.runner.close();panel.robot.close()
