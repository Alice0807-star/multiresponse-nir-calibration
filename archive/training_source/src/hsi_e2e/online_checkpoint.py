"""Strict, read-only loading of owned and explicitly hash-locked online states."""
import hashlib
from pathlib import Path
import numpy as np
import torch


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load_online_state(path, *, expected_sha256, expected_seed, expected_arm,
                      expected_context_sha256, expected_budget):
    # All calls concern our own experiment files. Verify identity before
    # deserialization; this is not a loader for arbitrary third-party pickles.
    if file_hash(path)!=expected_sha256:
        raise ValueError('Checkpoint hash mismatch')
    packed=torch.load(path,weights_only=False,map_location='cpu')
    if set(packed)!={'search','torch_rng'}:
        raise ValueError('Checkpoint schema mismatch')
    s=packed['search'];ev=s.ev
    if (s.seed,s.arm,ev.context_hash,ev.budget)!=(expected_seed,expected_arm,expected_context_sha256,expected_budget):
        raise ValueError('Checkpoint seed/arm/data/budget mismatch')
    if not 0<ev.fit_count<=ev.budget or len(ev.known)!=len(ev.ledger) or len(ev.known)!=ev.fit_count:
        raise ValueError('Paid fit/cache/ledger mismatch')
    if [r['actual_fits'] for r in ev.ledger]!=list(range(1,ev.fit_count+1)):
        raise ValueError('Noncontinuous paid fit clock')
    if {tuple(r['subset']) for r in ev.ledger}!=set(ev.known):
        raise ValueError('Cache identity mismatch')
    for subset,r in ev.known.items():
        if len(subset)!=16 or tuple(sorted(set(subset)))!=subset or min(subset)<0 or max(subset)>=ev.context.n_bands:
            raise ValueError('Invalid actual selected bands')
        if not np.isfinite(r['r2']).all() or not np.isfinite(r['residual_cov']).all():
            raise ValueError('Nonfinite cached context')
    best=max(ev.known,key=lambda subset:ev.known[subset]['mean_r2'])
    if ev.known[best]['mean_r2']!=ev.known[ev.best_subset]['mean_r2']:
        raise ValueError('Incumbent mismatch')
    if s.pending is not None:
        raise ValueError('Failure-state or partial-batch checkpoint needs explicit diagnosis')
    if expected_arm in ['adaptive','frozen']:
        learner=s.learner
        if learner is None or len(learner.models)!=3 or learner.adaptive!=(expected_arm=='adaptive'):
            raise ValueError('Feedback ensemble/control mismatch')
        if learner.updates!=512+(ev.fit_count-800)//8:
            raise ValueError('Feedback update clock mismatch')
        for model,opt in zip(learner.models,learner.optimizers):
            if model.kind!='conditional_three_target_gain_attention_v1' or model.dimension!=32:
                raise ValueError('Conditional architecture mismatch')
            if any(p.requires_grad!=learner.adaptive for p in model.attention.parameters()):
                raise ValueError('Attention trainability mismatch')
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite model')
            expected={id(p) for p in model.parameters() if p.requires_grad}
            actual={id(p) for group in opt.param_groups for p in group['params']}
            if expected!=actual:
                raise ValueError('Optimizer does not reference this model')
        learner.audit()
    elif s.learner is not None:
        raise ValueError('Unexpected learner in unlearned control')
    if packed['torch_rng'].dtype!=torch.uint8 or packed['torch_rng'].ndim!=1:
        raise ValueError('Invalid torch RNG')
    return s,packed['torch_rng']
