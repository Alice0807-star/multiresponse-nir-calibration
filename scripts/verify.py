"""Verify this release; optional full no-fit checks require local provider data."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def run_script(name,*arguments):
    command=[sys.executable,'-X','utf8',str(ROOT/'scripts'/name),*map(str,arguments)]
    result=subprocess.run(command,cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
    if result.returncode:
        print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
        raise RuntimeError('Check failed: '+name+' (exit '+str(result.returncode)+')')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all',action='store_true',help='Also recalculate metrics, numerical model predictions and original fixed-prediction resampling.')
    p.add_argument('--figures',action='store_true',help='Also redraw ten figure pages; install requirements-figures.txt first.')
    a=p.parse_args()
    output=ROOT/'generated';output.mkdir(exist_ok=True)
    report={'requested_full_no_fit_checks':a.all,'stages':{},'new_fits':0,
            'full_training_reproduction_claimed':False,'independent_validation_claimed':False}
    try:
        manifest=json.loads((ROOT/'checksums/manifest.json').read_text(encoding='utf-8'))
        for name,record in manifest['files'].items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Missing/invalid distributed file: '+name)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Distributed file checksum mismatch: '+name)
        report['stages']['integrity']={'status':'PASSED','files':len(manifest['files'])}
        if a.all and not (ROOT/'data/local/reference_data.npz').is_file():
            raise FileNotFoundError('Full no-fit checks need data/local/reference_data.npz. Import the official CGL/Corn MAT or ZIP files first; see DATA_SOURCES.md.')
        if a.all:
            for script,label in [('recheck_metrics.py','metrics'),('recheck_models.py','numeric_model_replay'),('recheck_resampling.py','original_fixed_prediction_resampling')]:
                run_script(script);report['stages'][label]={'status':'PASSED'}
        else:
            for label in ['metrics','numeric_model_replay','original_fixed_prediction_resampling']:
                report['stages'][label]={'status':'NOT_REQUESTED','requires_local_source_data':True}
        run_script('recheck_evidence.py');report['stages']['archived_evidence']={'status':'PASSED'}
        # Default mode intentionally avoids any generated predictive-error results
        # left by a previous --all invocation: choose its reference layer explicitly.
        run_script('recheck_tables.py',*(['--use-recomputed'] if a.all else []))
        report['stages']['manuscript_tables']={'status':'PASSED','basis':'recomputed_metrics_and_archived_protocol' if a.all else 'archived_record_consistency'}
        if a.figures:
            run_script('figures/recreate_all.py');report['stages']['figure_redraw']={'status':'PASSED','scope':'Numerical/geometric recreation, not automatic cross-environment visual equivalence.'}
        report['passed']=True
        report['full_supported_no_fit_verification_passed']=bool(a.all)
        code=0
    except Exception as exc:
        report['passed']=False;report['full_supported_no_fit_verification_passed']=False
        report['error']=str(exc);code=2
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return code

if __name__=='__main__':raise SystemExit(main())
