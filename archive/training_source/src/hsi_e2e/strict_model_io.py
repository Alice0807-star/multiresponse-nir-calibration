"""Explicit, checked policy and full-state IO for trusted local research files.

Format2 snapshots deliberately fail in the legacy ReLU-only loader. Full
snapshots contain pickle and must only be loaded from trusted local sources.
Metadata checks prevent accidental semantic mismatch, not malicious forgery.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import copy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from torch import nn

from .shared_action import SharedActionQNetwork
from .leaky_state import LeakyStateQNetwork
from .swap_rl import SwapContext
from .training_snapshot import EV_FIELDS, EXCLUDED

FORMAT_VERSION=2
REWARD_KIND='uniform_mean_R2_improvement'
TRAINERS={
    'hsi_e2e.r2_reward.R2RewardTrainer': 'relu',
    'hsi_e2e.mixed_restart.MixedRestartTrainer': 'relu',
    'hsi_e2e.leaky_state.LeakyStateTrainer': 'leaky_relu',
    'hsi_e2e.guarded_online.GuardedMixedRestartTrainer': 'relu',
    'hsi_e2e.guarded_online.GuardedLeakyStateTrainer': 'leaky_relu',
}


class StrictModelIOError(ValueError):
    pass


@dataclass
class LoadedPolicy:
    model: nn.Module
    metadata: dict
    artifact_sha256: str


def _json_copy(value):
    def default(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False,default=default))


def _json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _check_finite_tree(value,path='optimizer'):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all():raise StrictModelIOError(f'Nonfinite state at {path}')
    elif isinstance(value,np.ndarray):
        if not np.isfinite(value).all():raise StrictModelIOError(f'Nonfinite state at {path}')
    elif isinstance(value,dict):
        for key,child in value.items():_check_finite_tree(child,f'{path}.{key}')
    elif isinstance(value,(list,tuple)):
        for index,child in enumerate(value):_check_finite_tree(child,f'{path}[{index}]')
    elif isinstance(value,(float,np.floating)) and not np.isfinite(value):
        raise StrictModelIOError(f'Nonfinite state at {path}')


def _validate_optimizer_state(state,learning_rate):
    _check_finite_tree(state)
    if not isinstance(state,dict) or not state.get('param_groups') or 'state' not in state:
        raise StrictModelIOError('Incomplete optimizer state')
    if any(group.get('lr')!=learning_rate for group in state['param_groups']):
        raise StrictModelIOError('Optimizer learning rate conflicts with registered training config')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_dict_sha256(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if not isinstance(value,torch.Tensor):raise StrictModelIOError('State dict must contain only tensors')
        x=value.detach().cpu().contiguous()
        if not torch.isfinite(x).all():raise StrictModelIOError(f'Nonfinite model tensor: {name}')
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def _trainer_class(name):
    if name not in TRAINERS:raise StrictModelIOError(f'Unregistered trainer: {name}')
    module,cls=name.rsplit('.',1)
    return getattr(importlib.import_module(module),cls)


def _check_expected(actual,expected,path='metadata'):
    if expected is None:return
    if isinstance(expected,dict):
        if not isinstance(actual,dict):raise StrictModelIOError(f'Expected mapping at {path}')
        for key,value in expected.items():
            if key not in actual:raise StrictModelIOError(f'Missing expected field: {path}.{key}')
            _check_expected(actual[key],value,f'{path}.{key}')
    elif actual!=expected:raise StrictModelIOError(f'Semantic mismatch at {path}: {actual!r} != {expected!r}')


def _validate_metadata(meta):
    try:
        if meta['schema']!='hsi_e2e.strict_model_io' or meta['schema_version']!=1:
            raise StrictModelIOError('Unknown metadata schema')
        model,trainer,data=meta['model'],meta['trainer'],meta['data']
        if model['architecture']!='shared_action_v1':raise StrictModelIOError('Unsupported model architecture')
        if int(model['n_bands'])<1 or model['state_dim']!=4*model['n_bands']+3:
            raise StrictModelIOError('Inconsistent state dimensions')
        if model['hidden']<1 or model['feature_dim']<1 or model['candidate_activation']!='relu':
            raise StrictModelIOError('Unsupported candidate representation')
        name=trainer['class']
        if name not in TRAINERS:raise StrictModelIOError(f'Unregistered trainer: {name}')
        if model['state_activation']!=TRAINERS[name]:raise StrictModelIOError('Trainer/activation mismatch')
        if model['state_activation']=='relu':
            if model['negative_slope'] is not None:raise StrictModelIOError('ReLU must not carry a negative slope')
        elif not (isinstance(model['negative_slope'],(int,float)) and 0.<=model['negative_slope']<1.):
            raise StrictModelIOError('Invalid LeakyReLU slope')
        if trainer['reward_kind']!=REWARD_KIND or trainer['gamma']!=1.:
            raise StrictModelIOError('Unsupported reward definition or discount')
        if trainer['horizon']<1 or not 0<data['k']<=model['n_bands'] or data['n_components']<1:
            raise StrictModelIOError('Invalid environment specification')
        config=trainer['config']
        if trainer.get('optimizer')!='torch.optim.Adam':raise StrictModelIOError('Unregistered optimizer')
        if config.get('reward_kind')!=trainer['reward_kind']:
            raise StrictModelIOError('Reward metadata/config mismatch')
        if config.get('value_mode')!='return':raise StrictModelIOError('Only mean-R2 return critics are registered')
        if config.get('bound_bootstrap') is not False:
            raise StrictModelIOError('Unregistered target-bound semantics')
        if 'state_activation' in config and config['state_activation']!=model['state_activation']:
            raise StrictModelIOError('Activation metadata/config mismatch')
        if model['state_activation']=='leaky_relu' and config.get('state_negative_slope')!=model['negative_slope']:
            raise StrictModelIOError('LeakyReLU metadata/config slope mismatch')
        for scope in ['training','probe']:
            digest=data['context_sha256'][scope]
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise StrictModelIOError('Missing explicit immutable data context')
        _json_copy(meta)
    except (KeyError,TypeError) as exc:
        raise StrictModelIOError(f'Incomplete strict metadata: {exc}') from exc


def _model_spec(net):
    if type(net) not in {SharedActionQNetwork,LeakyStateQNetwork}:
        raise StrictModelIOError('Unknown network implementation; explicit registration required')
    if not isinstance(net.state_encoder,nn.Sequential) or len(net.state_encoder)!=2:
        raise StrictModelIOError('Unknown global state encoder')
    layer,act=net.state_encoder
    if not isinstance(layer,nn.Linear):raise StrictModelIOError('Unknown state transform')
    if type(act) is nn.ReLU:kind,slope='relu',None
    elif type(act) is nn.LeakyReLU:kind,slope='leaky_relu',act.negative_slope
    else:raise StrictModelIOError('Unknown state activation')
    return dict(architecture='shared_action_v1',n_bands=net.n_bands,state_dim=layer.in_features,
        hidden=net.hidden,feature_dim=int(net.band_features.shape[1]),state_activation=kind,
        negative_slope=slope,candidate_activation='relu')


def metadata_for_trainer(trainer):
    """Derive semantics from actual modules/config, refusing ambiguous rewards."""
    name=f'{type(trainer).__module__}.{type(trainer).__name__}'
    if name not in TRAINERS:raise StrictModelIOError(f'Unregistered trainer: {name}')
    if getattr(trainer,'architecture',None)!='shared_action':raise StrictModelIOError('Unknown trainer architecture')
    model=_model_spec(trainer.online)
    if type(trainer.optimizer) is not torch.optim.Adam:
        raise StrictModelIOError('Unregistered optimizer implementation')
    _validate_optimizer_state(trainer.optimizer.state_dict(),trainer.config['learning_rate'])
    if _model_spec(trainer.target)!=model:raise StrictModelIOError('Online/target architecture mismatch')
    if trainer.ev.n_bands!=model['n_bands'] or trainer.pev.n_bands!=model['n_bands']:
        raise StrictModelIOError('Evaluator/model dimensions mismatch')
    if trainer.ev.k!=trainer.pev.k or trainer.ev.n_components!=trainer.pev.n_components:
        raise StrictModelIOError('Train/probe regression settings mismatch')
    meta=dict(schema='hsi_e2e.strict_model_io',schema_version=1,model=model,
        trainer=dict(**{'class':name},reward_kind=trainer.config.get('reward_kind'),gamma=1.,optimizer='torch.optim.Adam',
            horizon=trainer.horizon,config=_json_copy(trainer.config)),
        data=dict(context_sha256=dict(training=trainer.ev.cache_context_sha256,probe=trainer.pev.cache_context_sha256),
            k=trainer.ev.k,n_components=trainer.ev.n_components))
    _validate_metadata(meta)
    return meta


def _construct_model(meta,state):
    spec=meta['model']
    if 'band_features' not in state:raise StrictModelIOError('Missing immutable candidate features')
    features=state['band_features'].detach().cpu().numpy()
    if features.shape!=(spec['n_bands'],spec['feature_dim']):raise StrictModelIOError('Candidate feature dimensions mismatch')
    if spec['state_activation']=='relu':
        net=SharedActionQNetwork(spec['state_dim'],spec['n_bands'],features,hidden=spec['hidden'])
    else:
        net=LeakyStateQNetwork(spec['state_dim'],spec['n_bands'],features,hidden=spec['hidden'],negative_slope=spec['negative_slope'])
    try:net.load_state_dict(state,strict=True)
    except RuntimeError as exc:raise StrictModelIOError(f'Model tensors do not match declared structure: {exc}') from exc
    if _model_spec(net)!=spec:raise StrictModelIOError('Reconstructed structure differs from declaration')
    return net


def _check_envelope(p,kind,expected=None):
    if p.get('format_version')!=FORMAT_VERSION or p.get('artifact_kind')!=kind:
        raise StrictModelIOError('Not a strict format2 artifact; legacy import must be explicit')
    meta=p.get('metadata');_validate_metadata(meta)
    if p.get('metadata_sha256')!=_json_sha(meta):raise StrictModelIOError('Metadata checksum mismatch')
    _check_expected(meta,expected)
    return meta


def _atomic_write(path,write):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    partial=path.with_name(path.name+'.partial')
    if partial.exists():raise FileExistsError(partial)
    write(partial);partial.rename(path)


def save_strict_checkpoint(trainer,path,*,checkpoint_label=None):
    meta=metadata_for_trainer(trainer)
    state={k:v.detach().cpu().clone() for k,v in trainer.online.state_dict().items()}
    p=dict(format_version=FORMAT_VERSION,artifact_kind='policy_checkpoint',metadata=meta,
        metadata_sha256=_json_sha(meta),state_dict=state,state_sha256=state_dict_sha256(state),
        checkpoint_label=_json_copy(checkpoint_label))
    before=torch.get_rng_state().clone()
    _atomic_write(path,lambda f:torch.save(p,f))
    if not torch.equal(before,torch.get_rng_state()):raise AssertionError('Checkpoint save consumed RNG')
    return dict(path=str(Path(path).resolve()),metadata=meta,artifact_sha256=file_sha256(path))


def load_strict_checkpoint(path,*,expected=None):
    before=torch.get_rng_state().clone()
    try:
        p=torch.load(path,map_location='cpu',weights_only=True)
        meta=_check_envelope(p,'policy_checkpoint',expected)
        if state_dict_sha256(p['state_dict'])!=p['state_sha256']:raise StrictModelIOError('Model checksum mismatch')
        net=_construct_model(meta,p['state_dict']);net.eval()
        return LoadedPolicy(net,copy.deepcopy(meta),file_sha256(path))
    finally:torch.set_rng_state(before)


def save_strict_training_state(trainer,path):
    """Preserve full process state; this format is not accepted by legacy IO."""
    before=torch.get_rng_state().clone();meta=metadata_for_trainer(trainer)
    def ev_state(ev):
        return dict(fields={k:getattr(ev,k) for k in EV_FIELDS if hasattr(ev,k)},
            fit_budget=ev.fit_budget,scope=getattr(ev,'scope',''),cache_context_sha256=ev.cache_context_sha256)
    online={k:v.detach().cpu().clone() for k,v in trainer.online.state_dict().items()}
    target={k:v.detach().cpu().clone() for k,v in trainer.target.state_dict().items()}
    p=dict(format_version=FORMAT_VERSION,artifact_kind='training_state',metadata=meta,
        metadata_sha256=_json_sha(meta),trainer_class=meta['trainer']['class'],
        attributes={k:v for k,v in trainer.__dict__.items() if k not in EXCLUDED},
        active_age=time.perf_counter()-trainer.started,online=online,target=target,
        online_sha256=state_dict_sha256(online),target_sha256=state_dict_sha256(target),
        optimizer=trainer.optimizer.state_dict(),torch_rng=before,
        numpy_rng=trainer.rng.bit_generator.state,replay=list(trainer.replay),replay_capacity=trainer.replay.maxlen,
        training=ev_state(trainer.ev),probe=ev_state(trainer.pev),ledger=getattr(trainer.ev,'ledger',[]),
        A=trainer.context.A,R=trainer.context.R,features=trainer.online.band_features.detach().cpu().numpy(),
        model_training=trainer.online.training)
    def write(f):
        with gzip.open(f,'xb',compresslevel=1) as stream:pickle.dump(p,stream,protocol=5)
    _atomic_write(path,write)
    if not torch.equal(before,torch.get_rng_state()):raise AssertionError('Snapshot save consumed RNG')
    return dict(path=str(Path(path).resolve()),metadata=meta,artifact_sha256=file_sha256(path),
        training_fits=trainer.ev.fit_count,probe_fits=trainer.pev.fit_count,updates=trainer.updates,
        replay_length=len(trainer.replay),episode=trainer.episode,step=trainer.step)


def load_strict_training_state(path,evaluator_factory,*,output_dir,expected=None):
    """Resume into an explicitly new directory, with no constructor regression queries."""
    before=torch.get_rng_state().clone();committed=False
    try:
        with gzip.open(path,'rb') as stream:p=pickle.load(stream)
        meta=_check_envelope(p,'training_state',expected)
        if p['trainer_class']!=meta['trainer']['class']:raise StrictModelIOError('Trainer declaration mismatch')
        attrs=p['attributes']
        if attrs['config']!=meta['trainer']['config']:raise StrictModelIOError('Training config differs from metadata')
        if attrs['horizon']!=meta['trainer']['horizon'] or attrs.get('architecture')!='shared_action':
            raise StrictModelIOError('Training environment differs from metadata')
        if p['numpy_rng']['bit_generator']!='PCG64':raise StrictModelIOError('Unsupported sampling RNG')
        _validate_optimizer_state(p['optimizer'],meta['trainer']['config']['learning_rate'])
        for key in ['online','target']:
            if state_dict_sha256(p[key])!=p[key+'_sha256']:raise StrictModelIOError(f'{key} checksum mismatch')
        cls=_trainer_class(meta['trainer']['class']);t=cls.__new__(cls);t.__dict__.update(attrs)
        ledger=p['ledger']
        for name,key in [('ev','training'),('pev','probe')]:
            packed=p[key];ev=evaluator_factory(packed['fit_budget'],packed['scope'],ledger)
            if (ev.cache_context_sha256!=meta['data']['context_sha256'][key] or
                    ev.cache_context_sha256!=packed['cache_context_sha256'] or
                    ev.n_bands!=meta['model']['n_bands'] or ev.k!=meta['data']['k'] or
                    ev.n_components!=meta['data']['n_components']):
                raise StrictModelIOError(f'{key} immutable data/settings mismatch')
            if ev.fit_count!=0:raise StrictModelIOError('Evaluator factory must not perform model fits')
            for field,value in packed['fields'].items():setattr(ev,field,value)
            setattr(t,name,ev)
        t.online=_construct_model(meta,p['online']);t.target=_construct_model(meta,p['target'])
        if not np.array_equal(p['features'],t.online.band_features.detach().numpy()):
            raise StrictModelIOError('Saved candidate features inconsistent')
        t.online.train(p['model_training']);t.target.eval()
        t.optimizer=torch.optim.Adam(t.online.parameters(),lr=t.config['learning_rate']);t.optimizer.load_state_dict(p['optimizer'])
        t.rng=np.random.default_rng(t.seed);t.rng.bit_generator.state=p['numpy_rng']
        t.replay=deque(p['replay'],maxlen=p['replay_capacity'])
        for attr,ev in [('context',t.ev),('probe_context',t.pev)]:
            setattr(t,attr,SwapContext(ev,hint_type='frozen_attention',correlation=np.zeros(ev.n_bands),
                A=p['A'],R=p['R'],horizon=t.horizon))
        if meta['trainer']['class'].startswith('hsi_e2e.guarded_online.'):
            restore=importlib.import_module('hsi_e2e.guarded_online').restore_guarded_runtime
            restore(t)
        elif meta['model']['state_activation']=='leaky_relu':
            t._pending_state_gradient_squares={};t._attach_state_gradient_hooks()
        _check_expected(metadata_for_trainer(t),meta)
        t.out=Path(output_dir);t.out.mkdir(parents=True,exist_ok=False)
        t.started=time.perf_counter()-p['active_age'];torch.set_rng_state(p['torch_rng']);committed=True
        return t
    finally:
        if not committed:torch.set_rng_state(before)


def declare_legacy_checkpoint_metadata(source_path,*,state_activation,trainer_class,reward_kind,
        data_context_sha256,k,n_components,horizon,negative_slope=None):
    """Explicit audited declaration for immutable historical weights.

    Dimensions can be checked from tensors; activation, reward, trainer and data
    context cannot and therefore MUST be supplied by the caller. This declaration
    records a caller assertion, not recovered provenance or proof of semantics.
    """
    p=torch.load(source_path,map_location='cpu',weights_only=True)
    if p.get('format_version')==FORMAT_VERSION:raise StrictModelIOError('Artifact is already strict')
    state=p['state_dict'];f=state['band_features'];w=state['state_encoder.0.weight']
    config=_json_copy(p.get('config',{}))
    if 'reward_kind' in config and config['reward_kind']!=reward_kind:
        raise StrictModelIOError('Explicit reward conflicts with legacy config')
    config['reward_kind']=reward_kind
    if 'state_activation' in config and config['state_activation']!=state_activation:
        raise StrictModelIOError('Explicit activation conflicts with legacy config')
    if state_activation=='leaky_relu':
        if 'state_negative_slope' in config and config['state_negative_slope']!=negative_slope:
            raise StrictModelIOError('Explicit slope conflicts with legacy config')
        config.update(state_activation=state_activation,state_negative_slope=negative_slope)
    elif negative_slope is not None:raise StrictModelIOError('ReLU declaration cannot carry Leaky slope')
    meta=dict(schema='hsi_e2e.strict_model_io',schema_version=1,
        model=dict(architecture='shared_action_v1',n_bands=int(f.shape[0]),feature_dim=int(f.shape[1]),
            state_dim=int(w.shape[1]),hidden=int(w.shape[0]),state_activation=state_activation,
            negative_slope=negative_slope,candidate_activation='relu'),
        trainer=dict(**{'class':trainer_class},reward_kind=reward_kind,gamma=1.,optimizer='torch.optim.Adam',horizon=int(horizon),config=config),
        data=dict(context_sha256=_json_copy(data_context_sha256),k=int(k),n_components=int(n_components)),
        provenance=dict(kind='explicit_legacy_declaration',source_path=str(Path(source_path).resolve()),
            source_sha256=file_sha256(source_path),semantics_recovered_from_tensors=False))
    _validate_metadata(meta)
    return meta


def import_legacy_checkpoint(source_path,destination_path,*,metadata):
    """Write a new strict copy; never rewrite the historical source."""
    _validate_metadata(metadata)
    prov=metadata.get('provenance',{})
    if prov.get('kind')!='explicit_legacy_declaration' or prov.get('source_sha256')!=file_sha256(source_path):
        raise StrictModelIOError('Missing or mismatched explicit legacy declaration')
    p=torch.load(source_path,map_location='cpu',weights_only=True)
    if p.get('format_version')==FORMAT_VERSION:raise StrictModelIOError('Source must be a legacy artifact')
    before=torch.get_rng_state().clone()
    try:
        _construct_model(metadata,p['state_dict'])
        state={k:v.detach().cpu().clone() for k,v in p['state_dict'].items()}
        packed=dict(format_version=FORMAT_VERSION,artifact_kind='policy_checkpoint',metadata=_json_copy(metadata),
            metadata_sha256=_json_sha(metadata),state_dict=state,state_sha256=state_dict_sha256(state),
            checkpoint_label=p.get('checkpoint'))
        _atomic_write(destination_path,lambda f:torch.save(packed,f))
    finally:torch.set_rng_state(before)
    return dict(path=str(Path(destination_path).resolve()),metadata=metadata,artifact_sha256=file_sha256(destination_path))
