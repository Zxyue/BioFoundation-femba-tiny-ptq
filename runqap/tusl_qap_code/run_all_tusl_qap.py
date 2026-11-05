#!/usr/bin/env python3
"""Run six fixed TUSL QAP precision recipes sequentially on one GPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from tusl_qap import sha,json_write

HERE=Path(__file__).resolve().parent
PRESETS={'W32A32':(0,0,None),'W4A32':(4,0,None),'W4A8':(4,8,99.9),'W8A32':(8,0,None),
         'W8A8':(8,8,99.9),'W4A4':(4,4,95.0)}


def worker_command(args,name,fold,seed,action):
    w,a,p=PRESETS[name]
    cmd=[sys.executable,'-u',str(HERE/'tusl_qap.py'),action,
         '--repo',str(args.repo),'--data-root',str(args.data_root),'--out',str(args.out/name),
         '--fold',str(fold),'--seed',str(seed),'--weight-bits',str(w),'--act-bits',str(a),
         '--act-percentile',str(p if p is not None else 99.9),
         '--pretrain-epochs',str(args.pretrain_epochs),'--finetune-epochs',str(args.finetune_epochs)]
    if args.retry_incomplete:cmd+=['--retry-incomplete']
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','run'])
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--configs',nargs='+',type=str.upper,choices=list(PRESETS),default=list(PRESETS))
    p.add_argument('--folds',type=int,nargs='+',choices=range(4),default=[0,1,2,3])
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--pretrain-epochs',type=int,default=50)
    p.add_argument('--finetune-epochs',type=int,default=10)
    p.add_argument('--skip-smoke',action='store_true',help='Only if the same setup has already passed GPU smoke.')
    p.add_argument('--retry-incomplete',action='store_true',help='Archive and rerun incomplete stages; completed stages remain verified and cached.')
    args=p.parse_args()
    for key in ('repo','data_root','out'):setattr(args,key,getattr(args,key).expanduser().resolve())
    for key in ('configs','folds','seeds'):
        if len(set(getattr(args,key)))!=len(getattr(args,key)):p.error(f'Duplicate --{key}.')
    if args.pretrain_epochs<1 or args.finetune_epochs<1 or any(not 0<=s<2**32 for s in args.seeds):
        p.error('Positive epochs and uint32 seeds are required.')
    args.out.mkdir(parents=True,exist_ok=True)
    work=[(name,fold,seed) for name in args.configs for seed in args.seeds for fold in args.folds]
    smoke=[(name,args.folds[0],args.seeds[0]) for name in args.configs] if not args.skip_smoke else []
    plan={'configs':{n:dict(zip(('weight_bits','act_bits','act_percentile'),PRESETS[n])) for n in args.configs},
          'folds':args.folds,'seeds':args.seeds,'formal_pipelines':len(work),'formal_training_jobs':2*len(work),
          'formal_test_jobs':len(work),'smoke_training_jobs':2*len(smoke),
          'smoke_commands':[worker_command(args,*job,'smoke') for job in smoke],
          'formal_commands':[worker_command(args,*job,'all') for job in work]}
    json_write(args.out/'batch_plan.json',plan)
    print(f'{len(work)} formal pipelines: {2*len(work)} training jobs + {len(work)} tests; {2*len(smoke)} short smoke training jobs.',flush=True)
    # Preflight all selected case plans before any GPU training starts.
    plans=args.out/'_plan_logs';plans.mkdir(exist_ok=True)
    for name,fold,seed in work:
        log=plans/f'{name}_fold{fold}_s{seed}.log'
        with log.open('w',encoding='utf-8') as f:
            subprocess.run(worker_command(args,name,fold,seed,'plan'),stdout=f,stderr=subprocess.STDOUT,check=True)
        print(f'Plan checked: {name} fold{fold} seed{seed}',flush=True)
    core={}
    for n in ('run_train.py','models/FEMBA.py','models/weight_quant.py','models/act_quant.py',
              'tasks/pretrain_qap_task.py','tasks/finetune_qap_task.py','tasks/finetune_task.py',
              'datasets/hdf5_dataset.py','data_module/finetune_data_module.py','schedulers/cosine.py'):
        core[n]=sha(args.repo/n)
    for file in sorted((args.repo/'config').rglob('*.yaml')):
        core[file.relative_to(args.repo).as_posix()]=sha(file)
    helpers={p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}
    helpers.update({p.relative_to(HERE).as_posix():sha(p) for p in sorted((HERE/'config').rglob('*.yaml'))})
    versions={'python':sys.version}
    for name in ('torch','pytorch-lightning','mamba-ssm','causal-conv1d','hydra-core','omegaconf','numpy','h5py','torchmetrics','timm'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    settings={'repo':str(args.repo),'data_root':str(args.data_root),'presets':PRESETS,
              'pretrain_epochs':args.pretrain_epochs,'finetune_epochs':args.finetune_epochs,
              'source_sha256':core,'helper_sha256':helpers,'versions':versions}
    # Canonicalize tuples before comparison with a saved JSON manifest.
    settings=json.loads(json.dumps(settings))
    campaign=args.out/'campaign.json'
    if campaign.exists() and json.loads(campaign.read_text())!=settings:
        raise ValueError('Campaign source/environment/settings changed; choose a new --out.')
    if args.action=='plan':
        print('Plans checked; no training or testing executed.',flush=True);return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Run on a GPU node in the CUDA/Mamba environment.')
    json_write(campaign,settings)
    for i,(name,fold,seed) in enumerate(smoke,1):
        print(f'[{i}/{len(smoke)}] Smoke {name}',flush=True)
        subprocess.run(worker_command(args,name,fold,seed,'smoke'),check=True)
    for i,(name,fold,seed) in enumerate(work,1):
        print(f'[{i}/{len(work)}] Full {name} fold{fold} seed{seed}',flush=True)
        subprocess.run(worker_command(args,name,fold,seed,'all'),check=True)
    from summarize_all import summarize
    summarize(args.out,args.configs,args.folds,args.seeds)
    print('Finished. Compare:',args.out/'comparison.csv',flush=True)


if __name__=='__main__':main()
