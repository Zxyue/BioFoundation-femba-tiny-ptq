#!/usr/bin/env python3
"""Compare fixed precision settings without mixing fold variation and seed variation."""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
from eval_tusl_qap import metrics

from run_all_tusl_qap import PRESETS as SPECS
KEYS=('macro_auroc_ovr','macro_ap_ovr','balanced_accuracy','macro_f1','accuracy')


def csv_write(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(out,configs,folds,seeds):
    out=Path(out);fold_rows=[];group_rows=[];details={};data_hashes={}
    for name in configs:
        w,a,p=SPECS[name]
        for seed in seeds:
            all_probs=[];all_labels=[];events=set();subjects=set();members=[]
            for fold in folds:
                root=out/name/f'fold{fold}'/f'seed{seed}'/'test'
                r=json.loads((root/'metrics.json').read_text())
                if (r['fold'],r['seed'],r['weight_bits'],r['act_bits'])!=(fold,seed,w,a):
                    raise ValueError('Mismatched result configuration.')
                if a and not np.isclose(r['act_percentile'],p,rtol=0,atol=1e-9):raise ValueError('Mismatched percentile.')
                if r.get('test_activation_calibration') is not False:raise ValueError('Unexpected test-time calibration.')
                digest=r['test_h5_sha256']
                if fold in data_hashes and data_hashes[fold]!=digest:raise ValueError('Methods used different test data.')
                data_hashes[fold]=digest
                with np.load(root/'predictions.npz',allow_pickle=False) as d:
                    names=d['event'].tolist();subs={n.split('_',1)[0] for n in names}
                    if len(names)!=len(set(names)) or events&set(names) or subjects&subs:
                        raise ValueError('Duplicate held-out test events/subjects across folds.')
                    events.update(names);subjects.update(subs)
                    all_probs.append(d['event_probabilities']);all_labels.append(d['event_labels'])
                row={'config':name,'seed':seed,'fold':fold,'weight_bits':w,'act_bits':a,'act_percentile':p,
                     'n_events':r['events']['n'],'n_windows':r['windows']['n']}
                for level in ('events','windows'):
                    row.update({level+'_'+k:r[level][k] for k in KEYS})
                fold_rows.append(row);members.append(r)
            row={'config':name,'seed':seed,'weight_bits':w,'act_bits':a,'act_percentile':p,
                 'n_folds':len(folds),'n_test_subjects':len(subjects),'n_test_events':len(events)}
            for key in KEYS:
                values=[r['events'][key] for r in members]
                row[key+'_fold_mean']=statistics.mean(values)
                row[key+'_fold_std']=statistics.stdev(values) if len(values)>1 else None
            pooled=metrics(np.concatenate(all_probs),np.concatenate(all_labels))
            group_rows.append(row)
            details[name+f'_seed{seed}']={'folds':list(folds),'fold_summary':row,'pooled_held_out_events':pooled}
    csv_write(out/'results_by_fold.csv',fold_rows)
    csv_write(out/'comparison.csv',group_rows)
    seed_rows=[]
    for name in configs:
        rows=[r for r in group_rows if r['config']==name]
        row={'config':name,'n_seeds':len(rows),'n_folds_per_seed':len(folds)}
        for key in KEYS:
            values=[r[key+'_fold_mean'] for r in rows]
            row[key+'_seed_mean']=statistics.mean(values)
            row[key+'_seed_std']=statistics.stdev(values) if len(values)>1 else None
        seed_rows.append(row)
    csv_write(out/'comparison_across_seeds.csv',seed_rows)
    (out/'summary.json').write_text(json.dumps({'configs':list(configs),'folds':list(folds),'seeds':list(seeds),
        'note':'Fold std and seed std are separate; only the test-pool subjects have held-out predictions.',
        'details':details},indent=2,allow_nan=False),encoding='utf-8')
    return group_rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--configs',nargs='+',choices=list(SPECS),default=list(SPECS))
    p.add_argument('--folds',type=int,nargs='+',choices=range(4),default=[0,1,2,3])
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    a=p.parse_args();summarize(a.out,a.configs,a.folds,a.seeds)
