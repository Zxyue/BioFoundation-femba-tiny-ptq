"""CPU tests for activation-state compatibility, range restoration and W4A8 guards."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from eval_tusl_qap import validate_activation_state,validate_config
REPO=Path(os.environ.get('FEMBA_QAP_REPO',Path.cwd()))
spec=importlib.util.spec_from_file_location('original_act_quant',REPO/'models/act_quant.py')
act=importlib.util.module_from_spec(spec);spec.loader.exec_module(act)

class FakeBidirectionalBlock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bidirectional=True;self.bidirectional_strategy='add'

class GraphFixture(torch.nn.Module):
    """Tests graph installation/state only; does not implement or execute Mamba."""
    def __init__(self,head=True):
        super().__init__()
        self.pos_embed=torch.nn.Parameter(torch.zeros(1,2,4))
        self.patch_embed=torch.nn.Identity()
        self.mamba_blocks=torch.nn.ModuleList([FakeBidirectionalBlock() for _ in range(2)])
        self.norm_layers=torch.nn.ModuleList([torch.nn.LayerNorm(4) for _ in range(2)])
        self.classifier=torch.nn.Linear(4,3) if head else None

def config(abits=8,percentile=99.9):
    return {'qap_weight_bits':4,'qap_act_bits':abits,'qap_act_percentile':percentile,
            'task':{'_target_':'tasks.finetune_qap_task.QAPFinetuneTask'},
            'model':{'num_classes':3,'classification_type':'mcc','num_blocks':2,'embed_dim':35,'num_channels':22,'seq_length':1280}}

class Tests(unittest.TestCase):
    def make_graph_state(self,head):
        model=GraphFixture(head)
        aq=act.install_act_quant(model,8)
        for i,q in enumerate(aq.q.values()):
            q.set_range(0.25+i/10);q.enabled.fill_(True)
        state={'model.'+k:v.clone() for k,v in model.state_dict().items()}
        return model,aq,state
    def test_reject_a32_checkpoint_for_a8_experiment(self):
        with self.assertRaisesRegex(ValueError,'activation bits'):validate_config(config(0))
        validate_config(config(0),act_bits=0)
    def test_reject_other_percentile(self):
        with self.assertRaisesRegex(ValueError,'percentile'):validate_config(config(percentile=99.999))
    def test_pretrain_graph_has_25_enabled_activation_ranges(self):
        model,aq,state=self.make_graph_state(False)
        self.assertEqual(aq.count_enabled(),25)
        self.assertEqual(validate_activation_state(state,pretrain=True),25)
    def test_finetune_graph_restores_all_32_ranges(self):
        original,aq,state=self.make_graph_state(True)
        restored=GraphFixture(True)
        aq2=act.install_act_quant(restored,8)
        self.assertEqual(aq2.count_enabled(),0)
        restored.load_state_dict(original.state_dict(),strict=True)
        self.assertEqual(aq2.count_enabled(),32)
        with torch.no_grad(): self.assertEqual(aq.ranges(),aq2.ranges())
        self.assertTrue(all(q.bits==8 and not q.calibrating for q in aq2.q.values()))
        self.assertEqual(validate_activation_state(state),32)
    def test_loading_without_activation_installation_fails(self):
        original,_,_=self.make_graph_state(True)
        with self.assertRaises(RuntimeError):GraphFixture(True).load_state_dict(original.state_dict(),strict=True)
    def test_disabled_quantizer_is_rejected(self):
        _,_,state=self.make_graph_state(True)
        state['model.act_quant.q.input.enabled'].fill_(False)
        with self.assertRaisesRegex(ValueError,'Inactive'):validate_activation_state(state)
    def test_overflow_range_is_rejected(self):
        _,_,state=self.make_graph_state(True)
        state['model.act_quant.q.input.log_range'].fill_(1000)
        with self.assertRaisesRegex(ValueError,'Invalid learned'):validate_activation_state(state)
    def test_a8_state_rejected_when_requesting_a32(self):
        _,_,state=self.make_graph_state(True)
        with self.assertRaisesRegex(ValueError,'Expected 0'):validate_activation_state(state,act_bits=0)
    def test_a4_graph_and_quantizer_state_roundtrip(self):
        model=GraphFixture(True);aq=act.install_act_quant(model,4)
        for q in aq.q.values():q.set_range(2.1);q.enabled.fill_(True)
        state={'model.'+k:v.clone() for k,v in model.state_dict().items()}
        self.assertEqual(validate_activation_state(state,act_bits=4),32)
        restored=GraphFixture(True);aq2=act.install_act_quant(restored,4)
        restored.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(aq2.count_enabled(),32)
        self.assertTrue(all(q.bits==4 for q in aq2.q.values()))
        x=torch.linspace(-4,4,51,requires_grad=True)
        y=aq2.q['input'](x);y.square().sum().backward()
        self.assertLessEqual(torch.unique(y.detach()).numel(),15)
        self.assertTrue(torch.isfinite(x.grad).all())
        c=config(abits=4,percentile=95)
        validate_config(c,weight_bits=4,act_bits=4,percentile=95)

    def test_real_a8_quantizer_forward_and_state_roundtrip(self):
        q=act.ActQuantizer(8);q.set_range(3.2);q.enabled.fill_(True)
        x=torch.linspace(-5,5,101,requires_grad=True)
        y=q(x);(y.square().mean()).backward()
        self.assertTrue(torch.isfinite(q.log_range.grad))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertLessEqual(float(y.detach().abs().max()),3.20001)
        other=act.ActQuantizer(8);other.load_state_dict(q.state_dict(),strict=True)
        self.assertTrue(torch.equal(y.detach(),other(x.detach())))

if __name__=='__main__':unittest.main(verbosity=2)
