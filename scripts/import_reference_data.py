"""Import user-obtained source MAT/ZIP files locally; never download or fit."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np

BASE = Path(__file__).resolve().parents[1]

def value_hash(array):
    a = np.ascontiguousarray(array,dtype='<f8')
    return hashlib.sha256(str(a.shape).encode('ascii')+a.tobytes()).hexdigest()

def read_mat(path):
    from scipy.io import loadmat
    path = Path(path)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as archive:
            members=[n for n in archive.namelist() if n.lower().endswith('.mat') and not n.startswith('__MACOSX/')]
            if len(members)!=1:raise ValueError('Expected one non-macOS MAT file in '+path.name)
            payload=io.BytesIO(archive.read(members[0]))
            return loadmat(payload,squeeze_me=True,struct_as_record=False)
    return loadmat(path,squeeze_me=True,struct_as_record=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cgl',required=True,type=Path,help='Official CGL_nir.mat or its ZIP')
    p.add_argument('--corn',required=True,type=Path,help='Official corn.mat or its ZIP')
    p.add_argument('--output',type=Path,default=BASE/'data/local/reference_data.npz')
    args=p.parse_args()
    expected=json.loads((BASE/'configs/data_sources.json').read_text(encoding='utf-8'))
    cm=read_mat(args.cgl); nm=read_mat(args.corn)
    arrays={'cgl_X':np.asarray(cm['Spectra'].data,dtype=np.float64),
            'cgl_y':np.asarray(cm['PropVals'].data,dtype=np.float64),
            'corn_X':np.asarray(nm['m5spec'].data,dtype=np.float64),
            'corn_y':np.asarray(nm['propvals'].data,dtype=np.float64)}
    checked={}
    for name,a in arrays.items():
        spec=expected['arrays'][name]
        if list(a.shape)!=spec['shape'] or not np.isfinite(a).all():
            raise ValueError('Source array shape/nonfinite mismatch: '+name)
        digest=value_hash(a)
        if digest!=spec['sha256_float64_le_shape_bytes']:
            raise ValueError('Source values/order differ from the frozen source: '+name)
        checked[name]={'shape':list(a.shape),'sha256':digest}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)
    report={'passed':True,'array_checks':checked,'source_rows_preserved':True,
            'CGL_order':'Spectra/PropVals original 231-row order, NOT Ycal then Ytest',
            'Corn_instrument':'m5','new_fits':0,
            'scope':'Local import only. Raw spectra and reference labels must not be committed.'}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'output':str(args.output),'new_fits':0}))

if __name__=='__main__':main()
