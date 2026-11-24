"""Re-evaluate the original fixed-prediction paired row resamples; no fitting.

This is limited to original A/B/C. It is not a full-training confidence interval,
and does not add resampling evidence for D or the new Corn partitions.
"""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def check(reference_data=None,output=None):
    reference_data=Path(reference_data) if reference_data else ROOT/'data/local/reference_data.npz'
    if not reference_data.is_file():
        raise FileNotFoundError('Local source reference values are required. Follow data/README.md; no reference labels are bundled.')
    index=json.loads((ROOT/'configs/evaluation_index.json').read_text(encoding='utf-8'))
    folder=ROOT/'evidence/fixed_prediction_resampling'
    expected=json.loads((folder/'expected.json').read_text(encoding='utf-8'))
    output=Path(output) if output else ROOT/'generated/resampling_report.json'
    max_error=0.;sources={}
    with np.load(reference_data,allow_pickle=False) as references:
        for source,n_rows in [('cgl',78),('corn',80)]:
            with np.load(folder/f'{source}_weights.npz',allow_pickle=False) as z:
                ids=z['sample_ids'].copy();counts=z['row_counts'].copy();arms=z['arms'].tolist()
            assert arms==['A','B','C'] and counts.shape==(10000,n_rows)
            assert ids.shape==(n_rows,) and len(set(ids.tolist()))==n_rows
            assert counts.dtype.kind in 'iu' and np.all(counts>=0) and np.all(counts.sum(1)==n_rows)
            y=references[source+'_y'][ids].copy()
            assert y.shape==(n_rows,4) and np.isfinite(y).all()
            predictions=[]
            for arm in arms:
                by_seed=[]
                for seed in [17,29,43]:
                    chosen=[p for p in index['pipelines'] if p['source']==source and p['cohort']=='original'
                            and int(p['selector_seed'])==seed and p['arm']==arm]
                    assert len(chosen)==(1 if source=='cgl' else 5)
                    pool={}
                    for row in chosen:
                        with np.load(ROOT/row['prediction_file'],allow_pickle=False) as z:
                            for sid,prediction in zip(z['ids'],z['prediction']):
                                assert int(sid) not in pool
                                pool[int(sid)]=prediction.copy()
                    assert set(pool)==set(ids.tolist())
                    by_seed.append(np.array([pool[int(sid)] for sid in ids],dtype=float))
                predictions.append(np.stack(by_seed))
            prediction=np.stack(predictions)
            assert prediction.shape==(3,3,n_rows,4)
            err2=(prediction-y[None,None,:,:])**2
            r2_batches=[];rmse_batches=[]
            for start in range(0,len(counts),500):
                w=counts[start:start+500].astype(float);n=w.sum(1)
                mu=w@y/n[:,None]
                variance=(w[:,:,None]*(y[None,:,:]-mu[:,None,:])**2).sum(1)/n[:,None]
                assert np.all(variance>0)
                mse=np.einsum('bn,msnt->bmst',w,err2,optimize=True)/n[:,None,None,None]
                r2_batches.append((1-mse/variance[:,None,None,:]).mean(2))
                rmse_batches.append(np.sqrt(mse).mean(2))
            r2=np.concatenate(r2_batches);rmse=np.concatenate(rmse_batches)
            macro={};targets=[]
            for contrast,c,a in [('B_minus_A',1,0),('C_minus_B',2,1),('C_minus_A',2,0)]:
                bounds=np.quantile((r2[:,c]-r2[:,a]).mean(1),[.025,.975])
                error=float(np.max(abs(bounds-np.asarray(expected['macro_r2'][source][contrast]))))
                max_error=max(max_error,error)
                assert error<=1e-12,(source,contrast,'macro R2',error)
                macro[contrast]=bounds.tolist()
                for record in expected['target_rmse']:
                    if record['source']!=source or record['contrast']!=contrast:continue
                    target=int(record['target'])
                    bound=np.quantile(rmse[:,c,target]-rmse[:,a,target],[.025,.975])
                    error=float(np.max(abs(bound-np.asarray(record['rmse_delta_percentile_interval']))))
                    max_error=max(max_error,error)
                    assert error<=1e-12,(source,contrast,target,'RMSE',error)
                    targets.append(dict(contrast=contrast,target=target,rmse_delta_percentiles=bound.tolist()))
            assert len(targets)==12
            sources[source]=dict(saved_row_resamples=10000,evaluation_rows=n_rows,
                                 macro_r2_percentiles=macro,target_rmse_percentiles=targets)
    report=dict(passed=True,sources=sources,max_absolute_error=max_error,new_model_fits=0,
                scope='Original A/B/C saved-prediction paired sensitivity only; no D/new-partition resampling, full-process coverage or independent validation claimed.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-data',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.reference_data,args.output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
