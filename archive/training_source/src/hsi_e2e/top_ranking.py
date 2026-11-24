"""One controlled change: vector MSE plus within-parent top-one supervision.

Inspired by listwise top-one learning (Cao et al., ICML2007). The hard winner
target and zero-gain reference are explicit adaptations, not original ListNet.
The reference exists in the training loss only; it introduces no search action,
reward clipping, task weighting, feature, architecture or acceptance change.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .conditional_feedback import actual_gain_feedback
from .online_conditional import FeedbackEnsemble

GAIN_SCALE = .01                 # Existing common R2 gain unit, unchanged.
TOP_WEIGHT = 1.                  # Fixed before any result; no weight search.
TIE_ATOL = 1e-10                 # Loss-only numerical ties, not a reward gate.


def top_one_feedback(prediction, true_gain, *, candidates=8):
    if prediction.shape != true_gain.shape or prediction.ndim != 3 or prediction.shape[1:] != (candidates, 3):
        raise ValueError('Top loss expects parent x8candidates x3targets')
    if prediction.shape[0] == 0 or not torch.isfinite(prediction).all() or not torch.isfinite(true_gain).all():
        raise FloatingPointError('Invalid/nonfinite top loss')
    # Only complete, already paid candidates provide teachers. Equal target mean
    # matches the unchanged search objective. No child label enters prediction.
    predicted = prediction.mean(-1)
    actual = true_gain.detach().mean(-1)
    zero = torch.zeros_like(predicted[:, :1])
    scores = torch.cat([zero, predicted], dim=1)/GAIN_SCALE
    actual_options = torch.cat([zero, actual], dim=1)
    best = actual_options.max(1, keepdim=True).values
    winners = torch.isclose(actual_options, best, atol=TIE_ATOL, rtol=0.)
    target = winners.to(prediction.dtype)/winners.sum(1, keepdim=True)
    loss = -(target*F.log_softmax(scores, dim=1)).sum(1).mean()
    return loss, dict(reference_teacher_fraction=float(target[:, 0].mean()),
                      positive_opportunity_fraction=float((actual.max(1).values > TIE_ATOL).float().mean()))


def combined_feedback(prediction, true_gain, *, top_weight=TOP_WEIGHT, candidates=8):
    if top_weight < 0 or not np.isfinite(top_weight):
        raise ValueError('Invalid top-loss coefficient')
    mse = actual_gain_feedback(prediction, true_gain)
    if top_weight == 0:
        # Exact legacy computation graph for the parity control, not MSE+0*CE.
        return mse, dict(vector_mse=float(mse.detach()), top_one_loss=0.,
                         reference_teacher_fraction=0., positive_opportunity_fraction=0.)
    grouped = prediction.reshape(-1, candidates, 3)
    teacher = true_gain.reshape(-1, candidates, 3)
    top, diagnostics = top_one_feedback(grouped, teacher, candidates=candidates)
    return mse+top_weight*top, dict(vector_mse=float(mse.detach()),
        top_one_loss=float(top.detach()), **diagnostics)


class TopFeedbackEnsemble(FeedbackEnsemble):
    """Same initialization, replay batches, optimizer and forward as legacy."""
    def __init__(self, warm_groups, *, adaptive, seeds=(13, 37, 73), top_weight=TOP_WEIGHT):
        super().__init__(warm_groups, adaptive=adaptive, seeds=seeds)
        self.top_weight = top_weight
        self.loss_history = []

    def update(self):
        # This sampler intentionally matches the locked legacy update exactly.
        ids = self.rng.integers(len(self.groups), size=16)
        samples = []
        for i in ids:
            group = self.groups[i]
            j = self.rng.choice(len(group['gain']), 8, replace=len(group['gain']) < 8)
            samples.append({k: v[j] for k, v in group.items()})
        q, t = self.tensors(np.concatenate([g['q'] for g in samples]), np.concatenate([g['t'] for g in samples]))
        y = torch.from_numpy(np.concatenate([g['gain'] for g in samples]).astype('float32'))
        losses, records = [], []
        for model, optimizer in zip(self.models, self.optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss, record = combined_feedback(model(q, t), y, top_weight=self.top_weight)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite top-feedback gradient')
            ag = sum(float(p.grad.detach().double().square().sum()) for p in model.attention.parameters() if p.grad is not None)**.5
            if self.adaptive and ag <= 0:
                raise RuntimeError('Top attention feedback disconnected')
            self.min_attention_gradient = min(self.min_attention_gradient, ag)
            gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.))
            self.max_gradient = max(self.max_gradient, gn)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite top-feedback parameters')
            losses.append(float(loss.detach()))
            records.append(record)
        self.last_loss = float(np.mean(losses))
        self.updates += 1
        self.loss_history.append(dict(update=self.updates, total=self.last_loss,
            **{k: float(np.mean([r[k] for r in records])) for k in records[0]}))

    def audit(self):
        base = super().audit()
        return dict(**base, training_loss='vector_MSE_plus_hard_top_one', top_weight=self.top_weight,
            top_loss_records=len(self.loss_history), last_components=self.loss_history[-1] if self.loss_history else None)
