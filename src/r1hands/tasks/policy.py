"""Small state-based task baseline; separate episodes/seeds are held out for evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from r1hands.tasks.episodes import EPISODE_SCHEMA

MODEL_SCHEMA='r1hands.task-policy.v1'
# Measured hand positions, object xyz/linear velocity, and support phase. No human landmarks.
INPUT_INDICES=np.array([0,1,2,3,4,5,12,13,14,19,20,21,25])


def train(paths,destination,include_failures=False,seed=0,ridge=0.1):
    rows=[];seen=set();metadata=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata_json']))
            if meta.get('schema')!=EPISODE_SCHEMA:raise ValueError('Expected task episodes')
            if meta['episode_id'] in seen:raise ValueError('Duplicate episode')
            seen.add(meta['episode_id'])
            if not meta['success'] and not include_failures:continue
            state,action,valid=data['observation_state'],data['action'],data['input_valid']
            if state.shape!=(len(action)+1,26) or action.shape[1:]!=(6,) or len(valid)!=len(action):
                raise ValueError('Episode state/action alignment is invalid')
            if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('Nonfinite episode')
            for i in np.flatnonzero(valid):rows.append((state[i,INPUT_INDICES],action[i],int(meta['seed']),str(path)))
            metadata.append(meta)
    if not rows:raise ValueError('No usable successful episodes; collect demonstrations or explicitly include failures')
    groups=np.unique([r[2] for r in rows])
    if len(groups)<2:raise ValueError('At least two different reset seeds are required for held-out evaluation')
    if len({m['model_sha256'] for m in metadata})!=1:raise ValueError('Task scene models differ across episodes')
    if len({json.dumps(m['config'],sort_keys=True) for m in metadata})!=1:raise ValueError('Task configurations differ across episodes')
    validation=np.random.default_rng(seed).permutation(groups)[:max(1,int(np.ceil(len(groups)*0.2)))]
    held=np.isin([r[2] for r in rows],validation)
    x,y=np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows])
    mean=x[~held].mean(0);scale=np.maximum(x[~held].std(0),1e-6)
    a=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    if not np.isfinite(ridge) or ridge<=0:raise ValueError('Ridge must be positive')
    reg=np.eye(a.shape[1])*ridge;reg[-1,-1]=0
    weights=np.linalg.solve(a[~held].T@a[~held]+reg,a[~held].T@y[~held])
    prediction=np.clip(a@weights,0,1)
    report={'schema':MODEL_SCHEMA,'input_indices':INPUT_INDICES.tolist(),'input':'hand state + object state + support phase',
            'train_seeds':sorted(set(int(r[2]) for i,r in enumerate(rows) if not held[i])),
            'validation_seeds':validation.tolist(),'train_samples':int((~held).sum()),'validation_samples':int(held.sum()),
            'validation_action_mae':float(np.abs(prediction[held]-y[held]).mean()),
            'train_datasets':sorted({r[3] for i,r in enumerate(rows) if not held[i]}),
            'validation_datasets':sorted({r[3] for i,r in enumerate(rows) if held[i]}),
            'include_failures':include_failures,'source_counts':{},'scene_sha256':metadata[0]['model_sha256'],
            'task_version':metadata[0]['task_version'],'task_config':metadata[0]['config'],'ridge':ridge,'seed':seed,
            'scope':'offline action prediction; evaluate closed-loop success separately; simulation only'}
    for meta in metadata:
        for key,count in meta.get('source_counts',{}).items():report['source_counts'][key]=report['source_counts'].get(key,0)+count
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,weights=weights,mean=mean,scale=scale,metadata_json=np.array(json.dumps(report)))
    destination.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    return report


def load_policy(path):
    with np.load(path,allow_pickle=False) as data:
        report=json.loads(str(data['metadata_json']))
        weights,mean,scale=(np.array(data[k],float) for k in ('weights','mean','scale'))
    if (report.get('schema')!=MODEL_SCHEMA or weights.shape!=(14,6) or mean.shape!=(13,) or scale.shape!=(13,)
        or not all(np.isfinite(a).all() for a in (weights,mean,scale)) or (scale<=0).any()):
        raise ValueError('Invalid state-based task policy')
    def predict(state):
        x=np.asarray(state,float)
        if x.shape!=(26,) or not np.isfinite(x).all():raise ValueError('Expected 26 finite task-state values')
        return np.clip(np.r_[(x[INPUT_INDICES]-mean)/scale,1]@weights,0,1)
    return predict,report


def evaluate(path,seeds,release_s=1.5,duration_s=5):
    from r1hands.sim.hand_task import HandTask, TaskConfig
    predict,meta=load_policy(path)
    if any(int(s) in meta['train_seeds'] for s in seeds):raise ValueError('Rollout evaluation must use held-out seeds')
    results=[];env=HandTask(TaskConfig(**meta['task_config']))
    try:
        if env.metadata()['model_sha256']!=meta['scene_sha256']:raise ValueError('Policy scene does not match evaluation scene')
        for seed in seeds:
            obs=env.reset(int(seed))
            for step in range(round(duration_s/env.config.control_dt)):
                if not env.support_removed and obs['time_s']>=release_s:env.release_support();obs=env.observe()
                obs=env.step(predict(obs['state']))
            results.append({'seed':int(seed),'success':obs['success'],'final_object_position':obs['object_position'].tolist()})
    finally:env.close()
    return {'rollouts':results,'success_rate':float(np.mean([r['success'] for r in results])),
            'release_s':release_s,'duration_s':duration_s,'simulation_only':True}


def main():
    parser=argparse.ArgumentParser(description='State-based grasp task baseline')
    commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('episodes',nargs='+',type=Path);fit.add_argument('--output',type=Path,required=True)
    fit.add_argument('--include-failures',action='store_true');fit.add_argument('--seed',type=int,default=0)
    score=commands.add_parser('evaluate');score.add_argument('model',type=Path);score.add_argument('--seeds',nargs='+',type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(train(args.episodes,args.output,args.include_failures,args.seed) if args.command=='train' else evaluate(args.model,args.seeds),indent=2))


if __name__=='__main__':main()
