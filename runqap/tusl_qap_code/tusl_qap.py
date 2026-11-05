#!/usr/bin/env python3
"""One TUSL fold: random-init configurable W4/W8 QAP pretrain -> 3-class fine-tune -> held-out test."""
import argparse
import hashlib
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def path_override(key,path):
    return key+'='+json.dumps(Path(path).as_posix())


def case_root(args):
    root = args.out / 'smoke' if args.action == 'smoke' else args.out
    return root/f'fold{args.fold}'/f'seed{args.seed}'


def training_command(args, stage):
    pre = stage == 'pretrain'
    directory = case_root(args)/stage
    smoke = args.action == 'smoke'
    epochs = 1 if smoke else (args.pretrain_epochs if pre else args.finetune_epochs)
    search = 'hydra.searchpath=[' + json.dumps('file://'+(HERE/'config').as_posix()) + ']'
    cmd=[sys.executable,'-u',str(args.repo/'run_train.py'),
         '+experiment=FEMBA_pretrain' if pre else '+experiment=FEMBA_finetune',
         'task=pretrain_qap_task' if pre else 'task=finetune_qap_task',
         'data_module=pretrain_tusl_events' if pre else 'data_module=finetune_tusl_events',
         search, f'tag=tusl_w{args.weight_bits or 32}a{args.act_bits or 32}_{stage}_fold{args.fold}_s{args.seed}', f'seed={args.seed}',
         'model.embed_dim=35','model.num_blocks=2','model.num_channels=22','model.seq_length=1280',
         f'+qap_weight_bits={args.weight_bits}',f'+qap_act_bits={args.act_bits}',
         f'+qap_act_percentile={args.act_percentile:g}',
         'gpus=1','trainer.devices=1','trainer.strategy=auto','+trainer.precision=32-true',
         'trainer.max_steps=-1',f'trainer.max_epochs={epochs}',
         f'batch_size={32 if pre else 16}','num_workers=0',
         'model_checkpoint.monitor=val_loss','model_checkpoint.mode=min',
         'model_checkpoint.save_top_k=1','model_checkpoint.save_last=False',
         'final_validate=False','final_test=False','resume=False',
         path_override('io.base_output_path',directory),
         path_override('io.checkpoint_dirpath',directory/'checkpoints'),
         '++pretrained_safetensors_path=null']
    if pre:
        cmd+=['pretrained_checkpoint_path=null','model.num_classes=0',
              '+qap_lr_schedule=cosine','+qap_warmup_epochs=2']
    else:
        cmd+=[path_override('pretrained_checkpoint_path',case_root(args)/'pretrain/final.ckpt'),
              'model.num_classes=3','model.classification_type=mcc',
              'finetuning.freeze_layers=False','+trainer.accumulate_grad_batches=1',
              'scheduler.warmup_epochs=2']
    if smoke:
        cmd+=['+trainer.limit_train_batches=6','+trainer.limit_val_batches=2','trainer.log_every_n_steps=1']
    return cmd


def test_command(args,checkpoint):
    return [sys.executable,'-u',str(HERE/'eval_tusl_qap.py'),
            '--repo',str(args.repo),'--fold-dir',str(args.data_root/f'fold{args.fold}'),
            '--checkpoint',str(checkpoint),
            '--pretrain-checkpoint',str(case_root(args)/'pretrain/final.ckpt'),
            '--out-dir',str(case_root(args)/'test'),'--batch-size','32',
            '--expected-weight-bits',str(args.weight_bits),
            '--expected-act-bits',str(args.act_bits),
            '--expected-act-percentile',str(args.act_percentile)]


def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def checkpoint_audit(path,args,pretrain):
    import torch
    from omegaconf import OmegaConf
    from eval_tusl_qap import validate_config,validate_activation_state
    if not OmegaConf.has_resolver('env'):
        OmegaConf.register_new_resolver('env',lambda key:os.environ[key])
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    cfg=OmegaConf.create(ckpt['hyper_parameters']);state=ckpt['state_dict']
    validate_config(cfg,pretrain=pretrain,weight_bits=args.weight_bits,
                    act_bits=args.act_bits,percentile=args.act_percentile)
    n_act=validate_activation_state(state,pretrain=pretrain,act_bits=args.act_bits)
    scale_keys=[k for k in state if k.endswith('.parametrizations.weight.0.log_scale')]
    expected=(21 if pretrain else 28) if args.weight_bits else 0
    if len(scale_keys)!=expected:raise ValueError('Wrong number of saved weight quantizers.')
    for key in scale_keys:
        prefix=key[:-len('log_scale')]
        if not bool(state[prefix+'enabled']) or not bool(state[prefix+'initialized']):
            raise ValueError('Uninitialized/disabled saved weight quantizer.')
        scale=state[key].exp()
        if not bool(torch.isfinite(scale).all() and (scale>0).all()):
            raise ValueError('Invalid learned weight scale.')
    if int(cfg.seed)!=args.seed:raise ValueError('Wrong checkpoint seed.')
    budget=1 if args.action=='smoke' else (args.pretrain_epochs if pretrain else args.finetune_epochs)
    if int(cfg.trainer.max_epochs)!=budget:raise ValueError('Wrong checkpoint epoch budget.')
    epoch=int(ckpt.get('epoch',-1));steps=int(ckpt.get('global_step',-1))
    epoch_ok = (epoch in (budget - 1, budget)) if pretrain else (0 <= epoch < budget)
    if steps < 1 or not epoch_ok:
        raise ValueError('Expected fully trained pretraining final.ckpt or valid best fine-tuning checkpoint.')
    return {'epoch':epoch,'global_step':steps,'n_weight_quantizers':expected,'n_act_quantizers':n_act}


def stage_inputs(args,stage):
    root=case_root(args);inputs={}
    if stage in ('finetune','test'):
        inputs['pretraining']=sha(root/'pretrain/final.ckpt')
    if stage=='test':
        ckpts=sorted((root/'finetune/checkpoints').rglob('*.ckpt'))
        if len(ckpts)!=1:raise ValueError('Expected one best fine-tuning checkpoint.')
        inputs['fine_tuning']=sha(ckpts[0])
    return inputs


def execute(args,cmd,stage):
    root=case_root(args).resolve();directory=root/stage;marker=directory/'complete.json'
    signature=hashlib.sha256(json.dumps(cmd,ensure_ascii=True).encode()).hexdigest()
    inputs=stage_inputs(args,stage)
    if marker.exists():
        receipt=json.loads(marker.read_text())
        if receipt['command_sha256']!=signature or receipt['input_sha256']!=inputs:
            raise ValueError(f'Completed stage inputs/command changed: {directory}')
        for relative,digest in receipt['outputs'].items():
            target=(root/relative).resolve()
            if not target.is_relative_to(root) or not target.is_file() or sha(target)!=digest:
                raise ValueError(f'Completed output changed/disappeared: {target}')
        print(f'SKIP verified completed stage: {directory}',flush=True);return
    if directory.exists() and any(directory.iterdir()):
        if not args.retry_incomplete:
            raise FileExistsError(f'Incomplete stage: {directory}; inspect its log, then rerun with --retry-incomplete if appropriate.')
        source=directory.resolve();archive=(root/'interrupted'/f'{stage}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}').resolve()
        # A retry only renames this campaign's stage directory; no recursive deletion.
        if source.parent!=root or not source.is_relative_to(root) or not archive.is_relative_to(root):
            raise ValueError('Archive path is outside the intended case directory.')
        archive.parent.mkdir(parents=True,exist_ok=True);source.rename(archive)
        print(f'Preserved incomplete stage at {archive}',flush=True)
    directory.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,DATA_PATH=str(args.data_root/f'fold{args.fold}'),
             CHECKPOINT_DIR=str(directory),WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
    log=directory/f'{stage}.log'
    json_write(directory/'command.json',{'argv':cmd,'cwd':str(args.repo),'DATA_PATH':env['DATA_PATH'],'CHECKPOINT_DIR':env['CHECKPOINT_DIR']})
    print(f'{stage}: log -> {log}',flush=True)
    with log.open('w',encoding='utf-8') as f:
        subprocess.run(cmd,cwd=args.repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    audit={}
    if stage=='pretrain':
        outputs=[directory/'final.ckpt']
        if not outputs[0].is_file():raise FileNotFoundError('Training did not finish the full pretraining budget.')
        audit=checkpoint_audit(outputs[0],args,True)
    elif stage=='finetune':
        outputs=sorted((directory/'checkpoints').rglob('*.ckpt'))
        if len(outputs)!=1:raise ValueError('Expected exactly one best-val-loss checkpoint.')
        audit=checkpoint_audit(outputs[0],args,False)
    else:
        outputs=[directory/'metrics.json',directory/'predictions.npz']
        metrics=json.loads(outputs[0].read_text())
        if metrics['weight_bits']!=args.weight_bits or metrics['act_bits']!=args.act_bits or metrics['fold']!=args.fold or metrics['seed']!=args.seed:
            raise ValueError('Test metrics do not match the planned configuration.')
    json_write(marker,{'command_sha256':signature,'input_sha256':inputs,
                      'outputs':{p.relative_to(root).as_posix():sha(p) for p in outputs},'audit':audit})
    print(f'{stage} complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','smoke','pretrain','finetune','test','all'])
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True,help='Directory containing fold0 ... fold3.')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fold',type=int,choices=range(4),default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pretrain-epochs',type=int,default=50)
    p.add_argument('--finetune-epochs',type=int,default=10)
    p.add_argument('--weight-bits',type=int,choices=[0,4,8],default=4)
    p.add_argument('--act-bits',type=int,choices=[0,4,8],default=8)
    p.add_argument('--retry-incomplete',action='store_true',help='Preserve incomplete stage in interrupted/, then rerun it.')
    p.add_argument('--act-percentile',type=float,choices=[95,99,99.9,99.99,99.999,100],default=99.9)
    args=p.parse_args()
    for key in ('repo','data_root','out'):setattr(args,key,getattr(args,key).expanduser().resolve())
    if args.pretrain_epochs<1 or args.finetune_epochs<1 or not 0<=args.seed<2**32:
        p.error('Epochs must be positive and seed must be a 32-bit unsigned integer.')
    fold_dir=args.data_root/f'fold{args.fold}'
    manifest=json.loads((fold_dir/'data-manifest.json').read_text())
    if manifest['fold']!=args.fold or manifest['num_classes']!=3 or manifest['classification_type']!='mcc':
        raise ValueError('Wrong TUSL fold manifest.')
    data_hashes={}
    for split in ('train','val','test'):
        path=fold_dir/f'{split}.h5'
        data_hashes[split]=sha(path)
        if data_hashes[split]!=manifest['data'][split]['sha256']:
            raise ValueError(f'{split}.h5 does not match its manifest.')
    source_files=['run_train.py','eval_femba.py','models/FEMBA.py','models/weight_quant.py','models/act_quant.py',
                  'tasks/pretrain_qap_task.py','tasks/finetune_qap_task.py','tasks/finetune_task.py']
    source_files += [p.relative_to(args.repo).as_posix() for p in sorted((args.repo/'config').rglob('*.yaml'))]
    sources={n:sha(args.repo/n) for n in source_files}
    plan={'fold':args.fold,'seed':args.seed,'data_sha256':data_hashes,'source_sha256':sources,
          'protocol':f'random-init W{args.weight_bits or 32}A{args.act_bits or 32} QAP on TUSL fold train; three-class fine-tune; held-out fold test',
          'weight_bits':args.weight_bits,'act_bits':args.act_bits,'act_percentile':args.act_percentile if args.act_bits else None,
          'helper_sha256':{n:sha(HERE/n) for n in ('tusl_qap.py','eval_tusl_qap.py')},
          'data_config_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in sorted((HERE/'config').rglob('*.yaml'))},
          'training_budget':{'pretrain_epochs':1 if args.action=='smoke' else args.pretrain_epochs,
                            'finetune_epochs':1 if args.action=='smoke' else args.finetune_epochs},
          'pretrain_command':training_command(args,'pretrain'),
          'finetune_command':training_command(args,'finetune')}
    root=case_root(args);root.mkdir(parents=True,exist_ok=True)
    plan_path=root/'execution_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Source/data/training settings changed; use a new --out.')
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if args.action=='plan':
        print(json.dumps(plan,indent=2));return
    if args.action in ('pretrain','all','smoke'):
        execute(args,plan['pretrain_command'],'pretrain')
        if not (root/'pretrain/final.ckpt').is_file():
            raise FileNotFoundError('Missing final.ckpt: inspect pretrain.log; the full budget may not have completed.')
    if args.action in ('finetune','all','smoke'):
        if not (root/'pretrain/final.ckpt').is_file():
            raise FileNotFoundError('Run matching pretraining first; no TUAR or FP32 weights are loaded.')
        execute(args,plan['finetune_command'],'finetune')
    if args.action in ('test','all'):
        checkpoints=sorted((root/'finetune/checkpoints').rglob('*.ckpt'))
        if len(checkpoints)!=1:
            raise ValueError(f'Expected one best-val-loss checkpoint; found {len(checkpoints)}.')
        execute(args,test_command(args,checkpoints[0]),'test')
    if args.action=='smoke':
        print('Smoke finished in its own directory; no full test metrics were produced.')


if __name__=='__main__':
    main()

