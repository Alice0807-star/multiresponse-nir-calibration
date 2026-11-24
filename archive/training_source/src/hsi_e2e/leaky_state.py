"""One-change state-activation screen; legacy models and training stay intact."""
from __future__ import annotations
import gzip
import math
import pickle
import numpy as np
import torch
from torch import nn
from .mixed_restart import MixedRestartTrainer
from .shared_action import SharedActionQNetwork
from .training_snapshot import load_training_state


def set_state_activation(net, negative_slope):
    slope=float(negative_slope)
    if not np.isfinite(slope) or not 0. <= slope < 1.:
        raise ValueError('Expected finite slope in [0,1)')
    # Parameter-free: preserves every initial tensor, optimizer reference and RNG.
    net.state_encoder[1]=nn.LeakyReLU(negative_slope=slope)


class LeakyStateQNetwork(SharedActionQNetwork):
    def __init__(self,*args,negative_slope=.01,**kwargs):
        super().__init__(*args,**kwargs)
        set_state_activation(self,negative_slope)


class LeakyStateTrainer(MixedRestartTrainer):
    def __init__(self,*args,negative_slope=.01,**kwargs):
        super().__init__(*args,**kwargs)
        self.state_negative_slope=float(negative_slope)
        for net in [self.online,self.target]:set_state_activation(net,self.state_negative_slope)
        self.config.update(state_activation='leaky_relu',state_negative_slope=self.state_negative_slope)
        self.state_gradient_rows=[]
        self._pending_state_gradient_squares={}
        self._attach_state_gradient_hooks()

    def _attach_state_gradient_hooks(self):
        for name,param in self.online.state_encoder[0].named_parameters():
            def observe(gradient,key=name):
                value=float(gradient.detach().double().square().sum())
                if not math.isfinite(value):raise FloatingPointError('Nonfinite state-encoder gradient')
                self._pending_state_gradient_squares[key]=value
                # Returning None leaves the original gradient unmodified.
            param.register_hook(observe)

    def _learn(self,*args,**kwargs):
        previous=self.updates
        self._pending_state_gradient_squares={}
        super()._learn(*args,**kwargs)
        if self.updates>previous:
            if set(self._pending_state_gradient_squares)!={'weight','bias'}:
                raise AssertionError('Missing state-gradient observation')
            norm=math.sqrt(sum(self._pending_state_gradient_squares.values()))
            state=kwargs.get('state',args[0] if args else None)
            self.state_gradient_rows.append(dict(update=self.updates,transition=self.transitions,
                training_fit_count=self.ev.fit_count,phase='add' if state[-2] else 'remove',
                state_gradient_norm_before_clip=norm))
            # A long completely disconnected branch is a screen failure, not
            # a plateau to wait through. No trigger on a single zero batch.
            if len(self.state_gradient_rows)>=256 and all(r['state_gradient_norm_before_clip']==0.
                    for r in self.state_gradient_rows[-256:]):
                raise FloatingPointError('State branch has zero gradients for256 consecutive updates')


def load_leaky_training_state(path,evaluator_factory,*,output_dir=None):
    """Trusted local snapshots only; generic legacy loader assumes ReLU."""
    with gzip.open(path,'rb') as f:p=pickle.load(f)
    expected=f'{LeakyStateTrainer.__module__}.{LeakyStateTrainer.__name__}'
    if p['trainer_class']!=expected or p['attributes']['config'].get('state_activation')!='leaky_relu':
        raise ValueError('Not a LeakyStateTrainer snapshot')
    t=load_training_state(path,evaluator_factory,output_dir=output_dir,trainer_class=LeakyStateTrainer)
    for net in [t.online,t.target]:set_state_activation(net,t.state_negative_slope)
    t._pending_state_gradient_squares={}
    t._attach_state_gradient_hooks()
    return t
