import csv, hashlib, json, re
from pathlib import Path
from collections import Counter
import h5py,numpy as np
import argparse
parser=argparse.ArgumentParser(description='Read-only verification of TUSL raw events, HDF5 windows and subject folds.')
parser.add_argument('--splits',type=Path,required=True)
parser.add_argument('--raw',type=Path,required=True,help='TUSL root, v2.0.1 root, or edf root.')
parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args()
SPLIT=args.splits.resolve();RAW=args.raw.resolve()
for suffix in ('v2.0.1/edf','edf'):
    if (RAW/suffix).is_dir():RAW=RAW/suffix;break
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def txt(x):return x.decode('utf-8') if isinstance(x,bytes) else str(x)
folds=json.loads((SPLIT/'folds.json').read_text());declared=json.loads((SPLIT/'summary.json').read_text())
raw={};labelmap={'bckg':0,'seiz':1,'slow':2}
for path in sorted(RAW.rglob('*.csv')):
    if not re.search(r'_\d{2}$',path.stem):continue
    rows=list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#')))
    spans={(float(r['start_time']),float(r['stop_time']),r['label']) for r in rows}
    assert len(spans)==1,(str(path),spans)
    start,stop,label=next(iter(spans));subject=path.stem.split('_',1)[0]
    assert subject==path.relative_to(RAW).parts[0]
    assert path.stem not in raw
    recording=path.stem.rsplit('_',1)[0]
    assert (path.parent/(recording+'.edf')).is_file()
    raw[path.stem]={'subject':subject,'label':labelmap[label],'start':start,'stop':stop,'recording':recording}
result={'raw_root':str(RAW),'split_root':str(SPLIT),'raw_events':len(raw),'raw_subjects':len({r['subject'] for r in raw.values()}),'raw_label_counts':dict(Counter(r['label'] for r in raw.values())),'protocol':folds['protocol'],'folds':[]}
canonical_events=None;canonical_windows={};test_subjects=[];covered_events=set()
for fold in folds['folds']:
    fi=fold['fold'];root=SPLIT/f'fold{fi}';manifest=json.loads((root/'data-manifest.json').read_text())
    assert manifest['num_classes']==3 and manifest['classification_type']=='mcc'
    fr={'fold':fi,'splits':{}};subjects={};events={};recordings={};all_sources=set();fold_events=set()
    for part in ('train','val','test'):
        path=root/(part+'.h5');names=[];labels=[];signals={}
        assert path.stat().st_size==manifest['data'][part]['bytes']
        assert digest(path)==manifest['data'][part]['sha256'],(fi,part,'hash mismatch')
        with h5py.File(path,'r') as f:
            for group in f.values():
                assert set(('X','y','source_file'))<=set(group.keys())
                assert group['X'].shape[1:]==(22,1280)
                assert group['y'].shape==(len(group['X']),)
                n=[txt(x) for x in group['source_file'][:]];y=group['y'][:]
                assert len(n)==len(y)==len(group['X'])
                for i,(name,label) in enumerate(zip(n,y)):
                    assert name not in names
                    event,window=name.rsplit('_w',1);assert window in ('0','1') and event in raw
                    assert int(label)==raw[event]['label']
                    x=group['X'][i];assert np.isfinite(x).all()
                    h=hashlib.sha256(x.tobytes()).hexdigest()
                    if name in canonical_windows:assert canonical_windows[name]==h
                    else:canonical_windows[name]=h
                names+=n;labels+=y.tolist()
        assert len(names)==len(set(names))
        part_events={n.rsplit('_w',1)[0] for n in names}
        assert len(names)==2*len(part_events)
        for e in part_events:assert {e+'_w0',e+'_w1'}<=set(names)
        subjects[part]={raw[e]['subject'] for e in part_events};events[part]=part_events
        recordings[part]={raw[e]['recording'] for e in part_events}
        assert subjects[part]==set(fold[part]),(fi,part,'subject JSON differs from HDF5')
        assert not (all_sources&set(names));all_sources.update(names);fold_events|=part_events
        fr['splits'][part]={'subjects':len(subjects[part]),'recordings':len(recordings[part]),'events':len(part_events),'windows':len(names),'label_counts_windows':dict(Counter(labels)),'shape':[22,1280],'subject_ids':sorted(subjects[part]),'sha256_verified':True}
    fr['subject_overlap']={a+'_'+b:sorted(subjects[a]&subjects[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    fr['recording_overlap']={a+'_'+b:sorted(recordings[a]&recordings[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(fr['subject_overlap'].values()) and not any(fr['recording_overlap'].values())
    if canonical_events is None:canonical_events=fold_events
    else:assert fold_events==canonical_events
    assert set(folds['always_train'])<=subjects['train']
    assert not any(subjects['test']&s for s in test_subjects)
    test_subjects.append(subjects['test']);covered_events|=events['test']
    result['folds'].append(fr)
byrecord={}
for e,r in raw.items():byrecord.setdefault(r['recording'],[]).append(e)
conflicts=set()
for es in byrecord.values():
    for i,a in enumerate(es):
        for b in es[i+1:]:
            x,y=raw[a],raw[b]
            if x['label']!=y['label'] and x['start']<y['stop'] and y['start']<x['stop']:conflicts.update((a,b))
excluded=set(raw)-canonical_events
assert excluded==conflicts
result.update(usable_events=len(canonical_events),usable_windows=len(canonical_windows),usable_subjects=len({raw[e]['subject'] for e in canonical_events}),always_train_subjects=len(folds['always_train']),test_pool_subjects=len(set().union(*test_subjects)),test_pool_events=len(covered_events),excluded_conflicting_events=sorted(excluded),all_checks_passed=True,label_map=labelmap)
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
