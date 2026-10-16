"""Aligned simulation episodes: T actions and T+1 observed states/RGB frames."""
from __future__ import annotations

import json
from pathlib import Path
import uuid
import numpy as np

EPISODE_SCHEMA = 'r1hands.task-episode.v1'


class EpisodeRecorder:
    def __init__(self, env, directory, provenance=None):
        self.directory = Path(directory)
        self.episode_id = uuid.uuid4().hex
        self.metadata = {'schema': EPISODE_SCHEMA, 'episode_id': self.episode_id,
                         **env.metadata(), 'provenance': provenance or {}, 'events': []}
        self.observations, self.images, self.times = [], [], []
        self.targets, self.desired, self.sources, self.valid, self.ages, self.landmarks = [], [], [], [], [], []
        self.contact_flags, self.forces = [], []
        self.start_time = env.data.time
        env.success = False; env.hold_time = 0.0; env.hold_reference = None
        self.add_observation(env)

    def add_observation(self, env, replace=False):
        observation, rgb = env.observe(), env.rgb()
        values = (observation['state'].copy(), rgb, float(env.data.time - self.start_time),
                  np.array([name in observation['contacts']['fingers'] for name in ('thumb','index','middle','ring','pinky')]),
                  observation['contacts']['max_force_n'])
        for array, value in zip((self.observations,self.images,self.times,self.contact_flags,self.forces), values):
            if replace: array[-1] = value
            else: array.append(value)

    def release_event(self, env):
        self.metadata['events'].append({'event':'support_removed','observation_index':len(self.targets),
                                        'time_s':float(env.data.time-self.start_time)})
        self.add_observation(env, replace=True)

    def append(self, env, desired, source, valid, age, landmarks=None):
        self.targets.append(env.target.copy()); self.desired.append(np.asarray(desired,float).copy())
        self.sources.append(source); self.valid.append(bool(valid)); self.ages.append(float(age))
        self.landmarks.append(np.full((21,3),np.nan) if landmarks is None else np.asarray(landmarks,float).reshape(21,3))
        self.add_observation(env)

    def finish(self, env, reason='operator_finish', operator_outcome='unspecified'):
        if not self.targets:
            raise ValueError('An episode needs at least one action')
        self.directory.mkdir(parents=True,exist_ok=True)
        observation = env.observe()
        self.metadata.update(success=bool(env.success), termination_reason=reason,
                             operator_outcome=operator_outcome, steps=len(self.targets),
                             duration_s=self.times[-1], rgb_shape=list(self.images[0].shape),
                             final_object_position=observation['object_position'].tolist(),
                             alignment='observation[i] -> applied_action[i] -> observation[i+1]',
                             source_counts={s:self.sources.count(s) for s in set(self.sources)})
        path = self.directory / f'{self.episode_id}.npz'
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as file:
            np.savez_compressed(file, observation_state=np.asarray(self.observations,np.float32),
                                observation_rgb=np.asarray(self.images,np.uint8),time_s=np.array(self.times),
                                action=np.asarray(self.targets,np.float32),desired_action=np.asarray(self.desired,np.float32),
                                action_source=np.array(self.sources,dtype='U16'),input_valid=np.array(self.valid,bool),
                                input_age_s=np.array(self.ages,np.float32),human_landmarks=np.array(self.landmarks,np.float32),
                                contact_fingers=np.array(self.contact_flags,bool),max_contact_force_n=np.array(self.forces,np.float32),
                                metadata_json=np.array(json.dumps(self.metadata,sort_keys=True)))
        temporary.replace(path)
        path.with_suffix('.json').write_text(json.dumps(self.metadata,indent=2))
        return self.metadata
