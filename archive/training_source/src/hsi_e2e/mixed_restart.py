"""Change only complete-episode restart selection in the audited R2 trainer."""
from __future__ import annotations
import heapq
import json
import numpy as np
import torch
from .r2_reward import R2RewardTrainer,r2_cost


class MixedRestartTrainer(R2RewardTrainer):
    def __init__(self,*args,mix_probability=.5,elite_size=16,**kwargs):
        if not 0.<=mix_probability<=1. or elite_size<1:
            raise ValueError('Invalid restart mixture')
        super().__init__(*args,**kwargs)
        self.mix_probability=float(mix_probability);self.elite_size=int(elite_size)
        self.original_scored=tuple(self.scored)
        self.restart_rng=np.random.default_rng(self.seed+910000)
        self.restart_rows=[]
        self._schedule_restart(0)
        self.current,self.current_error=self.scored[0]
        self.episode_start_error=self.current_error

    def _schedule_restart(self,episode_index):
        index=episode_index%len(self.original_scored)
        # Keep the original cycle on the original branch. This separate RNG
        # never perturbs epsilon/replay sampling or frozen-policy probes.
        coin=float(self.restart_rng.random()) if self.mix_probability>0. else None
        use_elite=coin is not None and coin<self.mix_probability
        rank=None
        if use_elite:
            # Training cache ONLY: no probe/oracle/held-out diagnostic labels.
            elite=heapq.nsmallest(self.elite_size,self.ev._cache.items(),key=lambda pair:(pair[1],pair[0]))
            rank=int(self.restart_rng.integers(len(elite)))
            subset,error=elite[rank]
        else:
            elite=[];subset,error=self.original_scored[index]
        self.scored[index]=(tuple(subset),float(error))
        self.restart_rows.append(dict(episode=episode_index+1,source='elite' if use_elite else 'original',
            coin=coin,original_index=index,elite_rank=None if rank is None else rank+1,
            elite_candidates=len(elite),training_cache_size=len(self.ev._cache),
            training_fits=self.ev.fit_count,subset_json=json.dumps(subset),
            start_nmse=float(error),start_mean_r2=1.-r2_cost(self.ev,subset)))

    def _learn(self,*args,**kwargs):
        previous=self.updates
        super()._learn(*args,**kwargs)
        if self.updates>previous:
            if not all(np.isfinite(v) for v in self.td_trace[-1].values() if isinstance(v,(int,float))):
                raise FloatingPointError('Nonfinite TD/gradient statistics')
            if self.updates%256==0 and not all(torch.isfinite(v).all() for v in self.online.parameters()):
                raise FloatingPointError('Nonfinite online parameters')
        done=kwargs.get('done',args[5] if len(args)>5 else False)
        if done:
            # The inherited collector resets current from this next slot
            # immediately after the terminal learning call returns. No reward,
            # boundary, transition, optimizer or episode-length code is copied.
            self._schedule_restart(self.episode+1)
