"""Post-hoc development control: common bands, joint vs independent training.

No wavelength search, hyperparameter search, external labels, or old-file edits.
All 18 independent-model selections are sealed before this runner opens audit.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import math
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from hsi_e2e.framework_screen import metrics, nn_predict, advance_patience
from hsi_e2e.formal_runtime import read, write, sha, torch_atomic

OUT = ROOT / 'results/empirical_training_control_001'
OLD = ROOT / 'results/framework_confirmation_001'
SPLIT = 7117
SEEDS = (17, 29, 43)
ARMS = {'st_same_width': (64, 32), 'st_total_capacity': (24, 23)}
STOP = False


class ControlMLP(nn.Module):
    def __init__(self, inputs, outputs, widths=(64, 32)):
        super().__init__()
        a, b = widths
        self.shared = nn.Sequential(nn.Linear(inputs, a), nn.GELU(), nn.Dropout(.05),
                                    nn.Linear(a, b), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(b, 1) for _ in range(outputs)])

    def forward(self, x):
        state = self.shared(x)
        return torch.cat([head(state) for head in self.heads], dim=1)


def train(x, y, xv, yv, subset, seed, widths, *, fixed_epochs=None,
          max_epochs=1200, min_epochs=400, patience=300):
    """Same optimization/checkpoint rules as historical train_nn, generic output."""
    subset = list(subset)
    norm = dict(xm=x[:, subset].mean(0), xs=x[:, subset].std(0),
                ym=y.mean(0), ys=y.std(0))
    if min(norm['xs']) <= 1e-12 or min(norm['ys']) <= 1e-12:
        raise ValueError('Constant training column')
    xt = torch.from_numpy(((x[:, subset]-norm['xm'])/norm['xs']).astype('float32'))
    yt = torch.from_numpy(((y-norm['ym'])/norm['ys']).astype('float32'))
    torch.manual_seed(seed+720000)
    rng = np.random.default_rng(seed+730000)
    model = ControlMLP(len(subset), y.shape[1], widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, best, best_score, last_improvement = [], None, -float('inf'), 0
    patience_anchor, updates = -float('inf'), 0
    target = max_epochs if fixed_epochs is None else int(fixed_epochs)
    start = time.perf_counter()
    for epoch in range(1, target+1):
        if STOP:
            raise InterruptedError('Operator stop; do not continue automatically')
        model.train()
        for ids in np.array_split(rng.permutation(len(x)), math.ceil(len(x)/128)):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(xt[ids])-yt[ids])**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite parameter')
            updates += 1
        if fixed_epochs is not None:
            continue
        if epoch % 10 == 0 or epoch == target:
            score = metrics(yv, nn_predict(model, norm, xv, subset))
            history.append(dict(epoch=epoch, **score))
            if score['mean_r2'] > best_score:
                best_score = score['mean_r2']
                best = dict(state=copy.deepcopy(model.state_dict()), epoch=epoch, selection=score)
            patience_anchor, last_improvement = advance_patience(
                best_score, patience_anchor, last_improvement, epoch)
            if epoch >= min_epochs and epoch-last_improvement >= patience:
                break
    if fixed_epochs is not None:
        best = dict(state=copy.deepcopy(model.state_dict()), epoch=target, selection=None)
    if best is None:
        raise RuntimeError('No valid checkpoint')
    model.load_state_dict(best['state'])
    model.eval()
    return dict(model=model, norm=norm, subset=subset, epoch=best['epoch'],
                selection=best['selection'], history=history, epochs_run=epoch,
                updates=updates, seconds=time.perf_counter()-start)


def emit(message):
    line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line, flush=True)
    with (OUT/'run.log').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')


def inputs():
    files = [OLD/'protocol.json', OLD/'metrics.json', OLD/'summary.json',
             OLD/str(SPLIT)/'selection_data.npz', OLD/str(SPLIT)/'audit_data.npz',
             ROOT/'data/development_v1/development_only.npz']
    for seed in SEEDS:
        folder = OLD/str(SPLIT)/'models'/str(seed)/'greedy'
        files += [folder/name for name in ['selected.json', 'selected_model.pt',
                  'refit_model.pt', 'predictions.npz', 'audit.json']]
    return files


def prepare():
    if (OUT/'protocol.json').exists():
        return lock()
    if OUT.exists():
        raise RuntimeError('Existing folder without protocol')
    import run_framework_confirmation as prior
    prior.lock()
    import run_shared_warm_freeze as recent
    recent.lock()
    OUT.mkdir(parents=True)
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(), split=SPLIT,
             seeds=list(SEEDS), k=48, arms=ARMS,
             joint_reference='historical greedy same bands joint64/32 MLP',
             parameter_counts=dict(joint=5315, st_same_width_total=15747,
                                   st_total_capacity_total=5325),
             purpose='Separate shared-band selection from joint neural training;not a new method',
             scope='Post-hoc one historically adapted development split;3algorithmseeds,not independent replicates',
             selection_rule='Each independent target uses own reward R2 and own best epoch;no audit selection',
             optimizer='Unchanged AdamW .001/decay .001,batch128,dropout.05,clip5,min400,max1200,patience300,eval10',
             audit_rule='Seal all18 selections/checkpoints;refit on fit+reward for fixed per-target epoch;only then score336audit',
             decision_rule='Descriptive only;report mean and every target and every seed;no promotion from this single split',
             source_sha256={str(f):sha(f) for f in [Path(__file__),
                   ROOT/'tests/test_empirical_training_control.py']+sorted((ROOT/'src/hsi_e2e').glob('*.py'))},
             input_sha256={str(f):sha(f) for f in inputs()})
    write(OUT/'protocol.json', p)
    write(OUT/'status.json', dict(state='prepared', current_audit_opened=False))
    return p


def lock():
    p = read(OUT/'protocol.json')
    for key in ['source_sha256', 'input_sha256']:
        for file, digest in p[key].items():
            if sha(file) != digest:
                raise RuntimeError('Locked artifact changed: '+file)
    return p


def selected_paths():
    return [OUT/str(seed)/arm/str(task)/'selected.json'
            for seed in SEEDS for arm in ARMS for task in range(3)]


def seal():
    files = selected_paths()
    if any(not f.exists() for f in files):
        raise RuntimeError('Incomplete independent-model selections')
    files += [f.parent/'selected_model.pt' for f in files]
    signatures = {str(f):sha(f) for f in files}
    marker = OUT/'choices_locked.json'
    if marker.exists():
        if read(marker)['files'] != signatures:
            raise RuntimeError('Sealed choices changed')
    else:
        write(marker, dict(utc=datetime.now(timezone.utc).isoformat(), files=signatures,
                           independently_selected_models=18))


def main():
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__('STOP', True))
    p = prepare()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=1):
            with np.load(OLD/str(SPLIT)/'selection_data.npz', allow_pickle=False) as z:
                xf, yf, xr, yr = [z[k].copy() for k in ['X_fit','y_fit','X_reward','y_reward']]
                fit_ids, reward_ids = z['fit_ids'].copy(), z['reward_ids'].copy()
            if set(fit_ids)&set(reward_ids) or xf.shape!=(1008,641) or xr.shape!=(336,641):
                raise ValueError('Unexpected selection partition')
            write(OUT/'status.json', dict(state='selecting', current_audit_opened=False))
            for seed in SEEDS:
                subset = read(OLD/str(SPLIT)/'models'/str(seed)/'greedy/selected.json')['subset']
                for arm, widths in ARMS.items():
                    for task in range(3):
                        folder = OUT/str(seed)/arm/str(task)
                        if (folder/'selected.json').exists():
                            continue
                        trained = train(xf, yf[:,task:task+1], xr, yr[:,task:task+1],
                                        subset, seed, widths)
                        folder.mkdir(parents=True)
                        torch_atomic(folder/'selected_model.pt', dict(
                            state=trained['model'].state_dict(), norm=trained['norm'], subset=subset))
                        write(folder/'history.json', trained['history'])
                        row = {k:trained[k] for k in ['epoch','selection','epochs_run','updates','seconds']}
                        write(folder/'selected.json', dict(seed=seed, arm=arm, task=task,
                              subset=subset, widths=widths, **row))
                        emit(f'SELECT seed={seed} arm={arm} task={task} epoch={trained["epoch"]}')
            seal()
            lock()
            # Reference audit predictions are deliberately first opened after this global seal.
            with np.load(OLD/str(SPLIT)/'audit_data.npz', allow_pickle=False) as z:
                xa, ya, audit_ids = [z[k].copy() for k in ['X_audit','y_audit','audit_ids']]
            if set(audit_ids)&(set(fit_ids)|set(reward_ids)) or len(audit_ids)!=336:
                raise ValueError('Audit partition overlap')
            write(OUT/'status.json', dict(state='refitting_and_evaluating', current_audit_opened=True))
            x, y = np.vstack([xf,xr]), np.vstack([yf,yr])
            rows=[]
            for seed in SEEDS:
                reference = OLD/str(SPLIT)/'models'/str(seed)/'greedy'
                with np.load(reference/'predictions.npz', allow_pickle=False) as z:
                    if not np.array_equal(z['ids'],audit_ids) or not np.array_equal(z['target'],ya):
                        raise ValueError('Reference audit identity mismatch')
                    joint_pred = z['prediction'].copy()
                joint_score = metrics(ya,joint_pred)
                if not np.allclose(joint_score['r2'],read(reference/'audit.json')['audit']['r2'],
                                   atol=1e-14,rtol=0):
                    raise ValueError('Reference score mismatch')
                rows.append(dict(seed=seed,arm='joint_reference',audit=joint_score))
                for arm,widths in ARMS.items():
                    predictions=[];costs=[]
                    for task in range(3):
                        folder=OUT/str(seed)/arm/str(task)
                        chosen=read(folder/'selected.json')
                        trained=train(x,y[:,task:task+1],None,None,chosen['subset'],
                                      seed,widths,fixed_epochs=chosen['epoch'])
                        predictions.append(nn_predict(trained['model'],trained['norm'],xa,chosen['subset']))
                        torch_atomic(folder/'refit_model.pt',dict(state=trained['model'].state_dict(),
                                      norm=trained['norm'],subset=chosen['subset']))
                        write(folder/'refit.json',dict(epoch=trained['epoch'],updates=trained['updates'],
                              seconds=trained['seconds'],selection_seconds=chosen['seconds']))
                        costs.append(dict(selection_seconds=chosen['seconds'],refit_seconds=trained['seconds']))
                    pred=np.hstack(predictions)
                    folder=OUT/str(seed)/arm
                    np.savez_compressed(folder/'predictions.npz',prediction=pred,target=ya,ids=audit_ids)
                    row=dict(seed=seed,arm=arm,audit=metrics(ya,pred),training_costs=costs)
                    write(folder/'audit.json',row);rows.append(row)
                    emit(f'AUDIT seed={seed} arm={arm} meanR2={row["audit"]["mean_r2"]:.7f}')
            write(OUT/'metrics.json',rows)
            means={arm:dict(mean_r2=float(np.mean([r['audit']['mean_r2'] for r in rows if r['arm']==arm])),
                     r2=np.mean([r['audit']['r2'] for r in rows if r['arm']==arm],axis=0).tolist(),
                     rmse=np.mean([r['audit']['rmse'] for r in rows if r['arm']==arm],axis=0).tolist())
                   for arm in ['joint_reference',*ARMS]}
            contrasts={}
            for arm in ARMS:
                delta=np.array(means['joint_reference']['r2'])-means[arm]['r2']
                contrasts[arm]=dict(direction='joint minus independent',target_delta=delta.tolist(),
                    mean_delta=float(delta.mean()),seed_wins=sum(
                      next(r['audit']['mean_r2'] for r in rows if r['arm']=='joint_reference' and r['seed']==seed)>
                      next(r['audit']['mean_r2'] for r in rows if r['arm']==arm and r['seed']==seed)
                      for seed in SEEDS))
            caps={arm:sum(read(f)['epochs_run']==1200 for f in selected_paths() if f.parts[-3]==arm)
                  for arm in ARMS}
            write(OUT/'summary.json',dict(methods=means,joint_minus_st=contrasts,epoch_caps=caps,
                  scope=p['scope'],configuration_promoted=False))
            lock();seal()
            write(OUT/'status.json',dict(state='complete',current_audit_opened=True,
                                        new_selection_models=18,new_refits=18))
            write(OUT/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
                  metrics_sha256=sha(OUT/'metrics.json'),summary_sha256=sha(OUT/'summary.json')))
            emit('COMPLETE no new searches;no independent validation;no configuration promotion')
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        emit('STOPPED on problem;no automatic changes or reruns')
        raise


if __name__=='__main__':
    main()
