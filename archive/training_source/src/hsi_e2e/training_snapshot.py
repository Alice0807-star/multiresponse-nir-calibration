"""Complete, compressed snapshots for trusted local R2 training runs."""
from __future__ import annotations
from collections import deque
import gzip
import pickle
from pathlib import Path
import time
import numpy as np
import torch
from .r2_reward import R2RewardTrainer
from .shared_action import SharedActionQNetwork
from .swap_rl import SwapContext

EV_FIELDS = ['_cache','_prediction_cache','_fit_hint_cache','_r2_costs','_r2_variance',
    'total_calls','cache_hits','fit_count','empty_evaluations','fit_seconds','warning_messages']
EXCLUDED = {'ev','pev','online','target','optimizer','rng','replay','context','probe_context'}


def save_training_state(trainer, path):
    """No extra fitting or random draws. Immutable data are referenced by hash."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    torch_before=torch.get_rng_state().clone()
    def ev_state(ev):
        return dict(fields={k:getattr(ev,k) for k in EV_FIELDS if hasattr(ev,k)},
            fit_budget=ev.fit_budget,scope=getattr(ev,'scope',''),
            cache_context_sha256=ev.cache_context_sha256)
    payload=dict(format_version=1,trainer_class=f'{type(trainer).__module__}.{type(trainer).__name__}',
        attributes={k:v for k,v in trainer.__dict__.items() if k not in EXCLUDED},
        active_age=time.perf_counter()-trainer.started,
        online=trainer.online.state_dict(),target=trainer.target.state_dict(),
        optimizer=trainer.optimizer.state_dict(),torch_rng=torch_before,
        numpy_rng=trainer.rng.bit_generator.state,replay=list(trainer.replay),
        replay_capacity=trainer.replay.maxlen,training=ev_state(trainer.ev),probe=ev_state(trainer.pev),
        ledger=getattr(trainer.ev,'ledger',[]),A=trainer.context.A,R=trainer.context.R,
        features=trainer.online.band_features.detach().cpu().numpy(),
        model_training=trainer.online.training,
        cache_note='Hint caches are deterministic memoization and are rebuilt on load; model/replay/regression caches are retained')
    temp=path.with_name(path.name+'.partial')
    with gzip.open(temp,'xb',compresslevel=1) as f:pickle.dump(payload,f,protocol=5)
    temp.rename(path)
    if not torch.equal(torch_before,torch.get_rng_state()):
        raise AssertionError('Serialization consumed training RNG')
    return dict(training_fits=trainer.ev.fit_count,probe_fits=trainer.pev.fit_count,
        transitions=trainer.transitions,updates=trainer.updates,replay_length=len(trainer.replay),
        episode=trainer.episode,step=trainer.step,bytes=path.stat().st_size)


def load_training_state(path, evaluator_factory, *, output_dir=None, trainer_class=None):
    """Load only a trusted locally generated snapshot.

    evaluator_factory(fit_budget, scope, shared_ledger) must recreate the same
    immutable fit/reward data. Its data/settings hash is checked before restore.
    Construction performs no initial-pool regression queries.
    """
    with gzip.open(path,'rb') as f:p=pickle.load(f)
    if p['format_version']!=1:raise ValueError('Unknown snapshot format')
    if p['attributes'].get('mix_probability',0)>0 and trainer_class is None:
        raise ValueError('Mixed restart snapshots require the matching trainer_class')
    cls=R2RewardTrainer if trainer_class is None else trainer_class
    t=cls.__new__(cls)
    t.__dict__.update(p['attributes'])
    if output_dir is not None:
        t.out=Path(output_dir);t.out.mkdir(parents=True,exist_ok=False)
    ledger=p['ledger']
    for name,key in [('ev','training'),('pev','probe')]:
        packed=p[key]
        ev=evaluator_factory(packed['fit_budget'],packed['scope'],ledger)
        if ev.cache_context_sha256!=packed['cache_context_sha256']:
            raise ValueError('Snapshot immutable data/settings mismatch')
        for field,value in packed['fields'].items():setattr(ev,field,value)
        setattr(t,name,ev)
    B=t.ev.n_bands
    t.online=SharedActionQNetwork(4*B+3,B,p['features'])
    t.target=SharedActionQNetwork(4*B+3,B,p['features'])
    t.online.load_state_dict(p['online']);t.target.load_state_dict(p['target'])
    t.online.train(p['model_training']);t.target.eval()
    t.optimizer=torch.optim.Adam(t.online.parameters(),lr=t.config['learning_rate'])
    t.optimizer.load_state_dict(p['optimizer'])
    t.rng=np.random.default_rng(t.seed);t.rng.bit_generator.state=p['numpy_rng']
    t.replay=deque(p['replay'],maxlen=p['replay_capacity'])
    for attr,ev in [('context',t.ev),('probe_context',t.pev)]:
        setattr(t,attr,SwapContext(ev,hint_type='frozen_attention',correlation=np.zeros(B),
            A=p['A'],R=p['R'],horizon=t.horizon))
    t.started=time.perf_counter()-p['active_age']
    torch.set_rng_state(p['torch_rng'])
    return t
