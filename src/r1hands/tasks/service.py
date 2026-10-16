"""A separate process owns MuJoCo and its GL context. The panel never calls real hardware here."""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import queue
import struct
import threading
import time
import zlib

import numpy as np


def png_bytes(rgb):
    height,width,_ = rgb.shape
    def chunk(name,data):
        return struct.pack('!I',len(data))+name+data+struct.pack('!I',zlib.crc32(name+data)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in rgb)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,3))+chunk(b'IEND',b'')


def _jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:_jsonable(v) for k,v in value.items()}
    return value


def _worker(connection,directory,renderer):
    env=None
    try:
        # MuJoCo selects GL at import time. Configure only this child, before import,
        # so a server renderer does not change the panel's other simulator backends.
        if renderer != 'auto':os.environ['MUJOCO_GL']=renderer
        from r1hands.sim.hand_task import HandTask
        from r1hands.tasks.episodes import EpisodeRecorder
        env=HandTask();episode=None;saved=None
        def response():
            return {'observation':_jsonable(env.observe()),'recording':episode is not None,
                    'recorded_steps':len(episode.targets) if episode else 0,'saved':saved,
                    'png':png_bytes(env.rgb()),'config':env.metadata()}
        connection.send({'ok':True,**response()})
        while True:
            command,args=connection.recv()
            try:
                if command=='close':
                    if episode and episode.targets: episode.finish(env,'panel_shutdown')
                    break
                if command=='reset':
                    if episode:raise ValueError('Finish the episode before resetting')
                    env.reset(int(args.get('seed',0)))
                elif command=='release':
                    env.release_support()
                    if episode:episode.release_event(env)
                elif command=='begin':
                    if episode:raise ValueError('Episode already recording')
                    episode=EpisodeRecorder(env,directory,args.get('provenance'))
                elif command=='finish':
                    if not episode:raise ValueError('No active episode')
                    saved=episode.finish(env,operator_outcome=args.get('outcome','unspecified'));episode=None
                elif command=='tick':
                    obs=env.step(args['target'],args['valid'])
                    if episode:
                        episode.append(env,args['target'],args['source'],args['valid'],args['age'],args.get('landmarks'))
                        if obs['success'] or len(episode.targets)>=env.config.max_steps:
                            saved=episode.finish(env,'success' if obs['success'] else 'time_limit');episode=None
                connection.send({'ok':True,**response()})
            except Exception as error:
                connection.send({'ok':False,'error':str(error)})
    except BaseException as error:
        try:connection.send({'ok':False,'error':str(error)})
        except (BrokenPipeError,EOFError):pass
    finally:
        if env:env.close()
        connection.close()


class TaskService:
    def __init__(self,directory,renderer='auto'):
        if renderer not in ('auto','glfw','egl','osmesa'):raise ValueError('Unknown task renderer')
        self.renderer=renderer
        self.directory=Path(directory)
        self.started=False;self.error=None;self.latest=None;self.png=None
        self._queue=queue.Queue();self._stop=threading.Event();self._ready=threading.Event();self._lock=threading.Lock()
        self.target=np.zeros(6);self.source='manual';self.target_t=time.monotonic();self.landmarks=None
        self.mapping_enabled=False;self.locked=False

    def start(self):
        if not self.started:
            self.started=True
            self.thread=threading.Thread(target=self._run,daemon=True,name='task-simulation');self.thread.start()
        if not self._ready.wait(30):raise TimeoutError('Task simulation did not start within 30 seconds')
        if self.error:raise RuntimeError(self.error)
        return self.snapshot()

    def _accept(self,result):
        if not result.get('ok'):raise RuntimeError(result.get('error','Task process failed'))
        self.png=result.pop('png');self.latest=result

    def _run(self):
        context=mp.get_context('spawn')
        connection,child=context.Pipe()
        process=context.Process(target=_worker,args=(child,str(self.directory),self.renderer),daemon=True)
        try:
            process.start();child.close()
            if not connection.poll(25):raise TimeoutError('Task rendering initialization timed out')
            self._accept(connection.recv());self._ready.set()
            while not self._stop.is_set():
                start=time.monotonic()
                try:
                    command,args,answer=self._queue.get_nowait()
                except queue.Empty:
                    command,args,answer='tick',None,None
                if command=='tick':
                    with self._lock:
                        age=time.monotonic()-self.target_t
                        args={'target':self.target.tolist(),'source':self.source,
                              'valid':self.source not in ('mapping','hold') or (self.source=='mapping' and age<0.25),'age':age,'landmarks':self.landmarks}
                connection.send((command,args))
                if not connection.poll(20):raise TimeoutError('Task process stopped responding')
                result=connection.recv()
                if result.get('ok'):
                    self._accept(result)
                    if answer:answer.put((True,self.snapshot()))
                elif answer:answer.put((False,result.get('error')))
                else:raise RuntimeError(result.get('error'))
                self._stop.wait(max(0,0.05-(time.monotonic()-start)))
        except Exception as error:
            self.error=str(error);self._ready.set()
        finally:
            if process.is_alive():
                try:connection.send(('close',{}))
                except (BrokenPipeError,EOFError):pass
                process.join(3)
                if process.is_alive():process.terminate();process.join(2)
            connection.close()

    def call(self,command,**args):
        self.start()
        if command not in ('reset','release','begin','finish'):raise ValueError('Unknown task command')
        answer=queue.Queue(maxsize=1);self._queue.put((command,args,answer))
        try:ok,result=answer.get(timeout=25)
        except queue.Empty:raise TimeoutError(self.error or 'Task command timed out')
        if not ok:raise ValueError(result)
        if command=='reset':
            with self._lock:self.target=np.zeros(6);self.mapping_enabled=False;self.source='manual';self.landmarks=None
        return result

    def set_target(self,position,source='manual',landmarks=None,valid=True):
        q=np.asarray(position,float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Task needs six finite finger targets')
        with self._lock:
            if self.locked and source!='hold':return
            self.target=np.clip(q,0,1);self.source=source;self.target_t=time.monotonic() if valid else 0
            self.landmarks=None if landmarks is None else np.asarray(landmarks,float).reshape(21,3).tolist()

    def hold(self,lock=False):
        with self._lock:
            self.mapping_enabled=False;self.source='hold';self.locked=lock or self.locked;self.landmarks=None

    def snapshot(self):
        return {'started':self.started,'ready':self.latest is not None and self.error is None,'error':self.error,
                'renderer':self.renderer,'mapping_enabled':self.mapping_enabled,'locked':self.locked,**(self.latest or {})}

    def close(self):
        self._stop.set()
        if hasattr(self,'thread'):self.thread.join(5)
