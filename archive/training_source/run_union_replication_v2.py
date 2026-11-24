"""Versioned runtime repair;all mathematical experiment functions stay frozen."""
from pathlib import Path
import signal,warnings,traceback,sys
import torch
from threadpoolctl import threadpool_limits,threadpool_info
import run_union_replication as run
from hsi_e2e.formal_runtime import read,write,sha

ROOT=Path(__file__).resolve().parent
run.OUT=ROOT/'results/union_replication_002'
BRIDGE=ROOT/'results/union_thread_bridge_001'


def pin_and_record():
    torch.set_num_threads(4)
    if torch.get_num_threads()!=4:raise RuntimeError('Actual Torch4 not pinned')
    pools=threadpool_info()
    if any(p['num_threads']!=1 for p in pools if p['user_api']=='blas'):
        raise RuntimeError('Actual BLAS1 not pinned')
    return dict(actual_torch_threads=torch.get_num_threads(),actual_interop_threads=torch.get_num_interop_threads(),threadpools=pools)


def main():
    proof=read(BRIDGE/'verification.json')
    if not proof['passed'] or proof['models']!=21 or not proof['all_weights_exact']:
        raise RuntimeError('Exact21model bridge required')
    signal.signal(signal.SIGINT,lambda *_:setattr(run.pilot.base,'STOP',True))
    p=run.prepare()
    extra=[Path(__file__),ROOT/'verify_union_replication_v2.py',ROOT/'tests/test_union_runtime.py',ROOT/'check_union_thread_bridge.py']
    p['source_sha256'].update({str(f):sha(f) for f in extra})
    p['input_sha256'].update({str(BRIDGE/name):sha(BRIDGE/name) for name in ['protocol.json','verification.json','environment.json']})
    p['nn_hyperparameters']='Unchanged model/optimizer;explicit actualTorch4 inside global limits1;actualBLAS1,interop1 verified at runtime'
    p['runtime_repair']='First001startup stopped before0fits/0audit;explicit actual threads and exact21historical selections bridge;all experiment functions unchanged'
    p['runtime_entry']=str(Path(__file__));write(run.OUT/'protocol.json',p);run.lock()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error');environment=pin_and_record()
            environment.update(python_executable=sys.executable,torch_version=torch.__version__,numpy_version=run.np.__version__,
                               historical_selection_bridge_exact=True)
            write(run.OUT/'execution_environment.json',environment)
            write(run.OUT/'status.json',dict(state='selecting',current_audit_opened=False))
            panels=run.panels_all();pin_and_record()
            run.select_all(panels);pin_and_record()
            run.evaluate_all(panels);pin_and_record()
    except BaseException as error:
        state=read(run.OUT/'status.json')
        write(run.OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        write(run.OUT/'status.json',dict(state='stopped_on_problem',current_audit_opened=state.get('current_audit_opened',False)))
        run.emit('STOPPED on problem;no automatic changes or retries');raise

if __name__=='__main__':main()
