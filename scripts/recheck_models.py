"""Replay archived numerical PLS parameters; no fitting, pickle or Torch."""
from pathlib import Path
import argparse
import json
import numpy as np

BASE=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',type=Path,default=BASE/'data/local/reference_data.npz')
    p.add_argument('--output-root',type=Path,default=BASE/'generated')
    a=p.parse_args()
    output=a.output_root.resolve()
    protected=[BASE/x for x in ['scripts','results','configs','evidence','data','archive','figures','checksums']]
    if output==BASE or any(output==folder or output.is_relative_to(folder) for folder in protected):
        p.error('Output must be generated/ or a separate directory; repository inputs are protected.')
    if not a.references.is_file():p.error('Import official source files first; local reference_data.npz is missing.')
    index=json.loads((BASE/'results/models/index.json').read_text(encoding='utf-8'))
    with np.load(a.references,allow_pickle=False) as source, np.load(BASE/'results/models/final_models.npz',allow_pickle=False) as params:
        from import_reference_data import value_hash
        expected_arrays=json.loads((BASE/'configs/data_sources.json').read_text(encoding='utf-8'))['arrays']
        for name,spec in expected_arrays.items():
            if name.endswith('_X') and value_hash(source[name])!=spec['sha256_float64_le_shape_bytes']:
                raise ValueError('Imported source spectra values/order differ: '+name)
        errors=[];values=0
        for row in index['pipelines']:
            with np.load(BASE/row['prediction_file'],allow_pickle=False) as stored:
                ids=stored['ids'];expected=stored['prediction']
            x=source[row['source']+'_X'][ids]
            predicted=[]
            for model_id in row['models']:
                get=lambda key:params[model_id+'_'+key]
                subset=get('subset').astype(int)
                z=((x[:,subset]-get('xmean'))/get('xstd'))@get('coef').T+get('intercept')
                predicted.append(z*get('target_std')+get('target_mean'))
            actual=np.column_stack(predicted)
            if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Invalid replay: '+row['id'])
            error=float(np.max(np.abs(actual-expected)));values+=expected.size
            if not np.allclose(actual,expected,rtol=1e-12,atol=1e-12):
                raise AssertionError('Numeric PLS parameter replay differs: '+row['id']+' max='+str(error))
            errors.append(error)
    a.output_root.mkdir(parents=True,exist_ok=True)
    report={'passed':True,'pipeline_prediction_files':len(errors),'unique_numeric_models':index['unique_models'],
            'predicted_values_checked':values,'max_absolute_prediction_error':max(errors),
            'new_fits':0,'pickle_loading':False,'scope':'Numerical parameter replay of archived final PLS models; not retraining or independent validation.'}
    (a.output_root/'model_replay_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
