import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,f1_score,accuracy_score
import h5py
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import eval_tusl_qap as m

class Tests(unittest.TestCase):
    def test_multiclass_matches_sklearn(self):
        rng=np.random.default_rng(4);y=np.tile(np.arange(3),70)
        p=rng.dirichlet(np.ones(3),len(y));r=m.metrics(p,y)
        self.assertAlmostEqual(r['macro_auroc_ovr'],roc_auc_score(y,p,multi_class='ovr',average='macro'))
        self.assertAlmostEqual(r['macro_ap_ovr'],average_precision_score(np.eye(3)[y],p,average='macro'))
        self.assertAlmostEqual(r['macro_f1'],f1_score(y,p.argmax(1),average='macro'))
        self.assertAlmostEqual(r['balanced_accuracy'],balanced_accuracy_score(y,p.argmax(1)))
        self.assertAlmostEqual(r['accuracy'],accuracy_score(y,p.argmax(1)))
    def test_tied_scores(self):
        y=np.array([0,0,1,2,2,2]);p=np.full((6,3),1/3);r=m.metrics(p,y)
        self.assertAlmostEqual(r['macro_auroc_ovr'],0.5)
        self.assertAlmostEqual(r['macro_ap_ovr'],1/3)
    def test_event_mean_and_sorted_alignment(self):
        names=['bb_s1_t1_00_w1','aa_s1_t1_00_w0','bb_s1_t1_00_w0','aa_s1_t1_00_w1']
        p=np.array([[.1,.7,.2],[.8,.1,.1],[.2,.6,.2],[.6,.2,.2]]);y=np.array([1,0,1,0])
        n,l,v=m.aggregate_events(p,y,names)
        self.assertEqual(n.tolist(),['aa_s1_t1_00','bb_s1_t1_00']);self.assertEqual(l.tolist(),[0,1])
        np.testing.assert_allclose(v,[[.7,.15,.15],[.15,.65,.2]])
    def test_incomplete_event_rejected(self):
        with self.assertRaises(ValueError):m.aggregate_events([[1,0,0]],[0],['aa_w0'])
    def test_duplicate_event_window_rejected(self):
        with self.assertRaises(ValueError):m.aggregate_events([[1,0,0]]*2,[0,0],['aa_w0','aa_w0'])
    def test_conflicting_labels_rejected(self):
        with self.assertRaises(ValueError):m.aggregate_events([[1,0,0]]*2,[0,1],['aa_w0','aa_w1'])
    def test_wrong_checkpoint_settings_rejected(self):
        cfg={'qap_weight_bits':4,'qap_act_bits':8,'qap_act_percentile':99.9,'task':{'_target_':'tasks.finetune_qap_task.QAPFinetuneTask'},
             'model':{'num_classes':3,'classification_type':'mcc','num_blocks':2,'embed_dim':35,'num_channels':22,'seq_length':1280}}
        m.validate_config(cfg)
        cfg['qap_weight_bits']=8
        with self.assertRaises(ValueError):m.validate_config(cfg)
        cfg['qap_weight_bits']=4;cfg['model']['num_classes']=2
        with self.assertRaises(ValueError):m.validate_config(cfg)
    def test_binary_data_rejected(self):
        with self.assertRaises(ValueError):m.metrics(np.array([[.8,.1,.1],[.1,.8,.1]]),np.array([0,1]))

if __name__=='__main__':unittest.main(verbosity=2)
