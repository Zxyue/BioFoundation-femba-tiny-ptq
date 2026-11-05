"""Batch scheduling, precision guards, resume/retry and cross-method summary tests."""
import argparse,contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import tusl_qap as worker
from run_all_tusl_qap import PRESETS,worker_command
from eval_tusl_qap import validate_config,metrics
from summarize_all import summarize


def cfg(w,a,p,pre=False):
    return {'qap_weight_bits':w,'qap_act_bits':a,'qap_act_percentile':p,'seed':42,
            'task':{'_target_':'tasks.'+('pretrain_qap_task.QAPPretrainTask' if pre else 'finetune_qap_task.QAPFinetuneTask')},
            'model':{'num_classes':0 if pre else 3,'classification_type':'mcc','num_blocks':2,'embed_dim':35,'num_channels':22,'seq_length':1280},
            'trainer':{'max_epochs':50 if pre else 10}}

class Tests(unittest.TestCase):
    def args(self,root):
        return argparse.Namespace(repo=root/'repo',data_root=root/'data',out=root/'out',action='all',
               fold=0,seed=42,weight_bits=4,act_bits=8,act_percentile=99.9,
               pretrain_epochs=50,finetune_epochs=10,retry_incomplete=False)
    def command(self,args,phase,marker):
        root=worker.case_root(args);stage=root/phase
        output=stage/'final.ckpt' if phase=='pretrain' else stage/'checkpoints/run/epoch=0-step=1.ckpt'
        code='from pathlib import Path;'
        code+=f'p=Path({str(output)!r});p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"fake-test-checkpoint");'
        code+=f'm=Path({str(marker)!r});m.write_text(str(int(m.read_text())+1) if m.exists() else "1")'
        return [sys.executable,'-c',code]
    def test_all_six_configs_accepted_and_wrong_weight_rejected(self):
        for name,(w,a,p) in PRESETS.items():
            validate_config(cfg(w,a,p),weight_bits=w,act_bits=a,percentile=p or 99.9)
            with self.assertRaises(ValueError):
                validate_config(cfg(w,a,p),weight_bits=12-w,act_bits=a,percentile=p or 99.9)
    def test_batch_generates_24_distinct_formal_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.args(Path(temp));commands=[]
            for name in PRESETS:
                for fold in range(4):commands.append(worker_command(a,name,fold,42,'all'))
            self.assertEqual(len(commands),24)
            self.assertEqual(len({tuple(c) for c in commands}),24)
            for name,(w,ab,p) in PRESETS.items():
                cmd=worker_command(a,name,0,42,'all')
                self.assertEqual(cmd[cmd.index('--weight-bits')+1],str(w))
                self.assertEqual(cmd[cmd.index('--act-bits')+1],str(ab))
    def test_completed_stage_is_not_executed_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=self.args(root);a.repo.mkdir();counter=root/'counter'
            command=self.command(a,'pretrain',counter)
            with patch.object(worker,'checkpoint_audit',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                worker.execute(a,command,'pretrain');worker.execute(a,command,'pretrain')
            self.assertEqual(counter.read_text(),'1')
    def test_checkpoint_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=self.args(root);a.repo.mkdir();cmd=self.command(a,'pretrain',root/'counter')
            with patch.object(worker,'checkpoint_audit',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                worker.execute(a,cmd,'pretrain')
                (worker.case_root(a)/'pretrain/final.ckpt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'output changed'):worker.execute(a,cmd,'pretrain')
    def test_retry_preserves_incomplete_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=self.args(root);a.repo.mkdir();stage=worker.case_root(a)/'pretrain';stage.mkdir(parents=True)
            (stage/'partial.log').write_text('interrupted')
            cmd=self.command(a,'pretrain',root/'counter')
            with self.assertRaises(FileExistsError):worker.execute(a,cmd,'pretrain')
            a.retry_incomplete=True
            with patch.object(worker,'checkpoint_audit',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                worker.execute(a,cmd,'pretrain')
            archived=list((worker.case_root(a)/'interrupted').glob('pretrain_*/partial.log'))
            self.assertEqual(len(archived),1);self.assertEqual(archived[0].read_text(),'interrupted')
    def test_changed_parent_invalidates_completed_finetuning(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=self.args(root);a.repo.mkdir();pre=worker.case_root(a)/'pretrain/final.ckpt';pre.parent.mkdir(parents=True);pre.write_bytes(b'parent1')
            cmd=self.command(a,'finetune',root/'counter')
            with patch.object(worker,'checkpoint_audit',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                worker.execute(a,cmd,'finetune');pre.write_bytes(b'parent2')
                with self.assertRaisesRegex(ValueError,'inputs/command changed'):worker.execute(a,cmd,'finetune')
    def test_real_checkpoint_metadata_audit_all_stage_types(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=self.args(root)
            for name,(w,ab,p) in PRESETS.items():
                for pre in (True,False):
                    a.weight_bits=w;a.act_bits=ab;a.act_percentile=p or 99.9
                    state={}
                    for i in range((21 if pre else 28) if w else 0):
                        prefix=f'model.w{i}.parametrizations.weight.0.'
                        state[prefix+'log_scale']=torch.zeros(2)
                        state[prefix+'initialized']=torch.tensor(True);state[prefix+'enabled']=torch.tensor(True)
                    for i in range((25 if pre else 32) if ab else 0):
                        prefix=f'model.act_quant.q.q{i}.'
                        state[prefix+'log_range']=torch.zeros(())
                        state[prefix+'initialized']=torch.tensor(True);state[prefix+'enabled']=torch.tensor(True)
                    checkpoint={'state_dict':state,'hyper_parameters':cfg(w,ab,p,pre),'epoch':49 if pre else 2,'global_step':600 if pre else 100}
                    path=root/(name+str(pre)+'.ckpt');torch.save(checkpoint,path)
                    audit=worker.checkpoint_audit(path,a,pre)
                    self.assertEqual(audit['n_act_quantizers'],(25 if pre else 32) if ab else 0)
    def make_summary_fixture(self,root):
        for name,(w,a,p) in PRESETS.items():
            for fold in range(4):
                stage=root/name/f'fold{fold}/seed42/test';stage.mkdir(parents=True)
                labels=np.arange(3);probs=np.eye(3)*.97+.01
                events=np.array([f'f{fold}c{c}_s1_t1_00' for c in range(3)])
                r={'fold':fold,'seed':42,'weight_bits':w,'act_bits':a,'act_percentile':p,
                   'test_activation_calibration':False,'test_h5_sha256':f'same-fold-{fold}',
                   'events':metrics(probs,labels),'windows':metrics(np.repeat(probs,2,axis=0),np.repeat(labels,2))}
                (stage/'metrics.json').write_text(json.dumps(r))
                np.savez_compressed(stage/'predictions.npz',event=events,event_labels=labels,event_probabilities=probs)
    def test_summary_keeps_configs_folds_and_seeds_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_summary_fixture(root)
            rows=summarize(root,list(PRESETS),[0,1,2,3],[42])
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r['n_folds']==4 for r in rows))
            self.assertTrue((root/'comparison.csv').exists())
            self.assertIn('seed_std',(root/'comparison_across_seeds.csv').read_text(encoding='utf-8-sig'))
    def test_summary_rejects_different_data_across_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.make_summary_fixture(root)
            f=root/'W8A8/fold0/seed42/test/metrics.json';r=json.loads(f.read_text());r['test_h5_sha256']='other';f.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'different test data'):summarize(root,list(PRESETS),[0,1,2,3],[42])

if __name__=='__main__':unittest.main(verbosity=2)
