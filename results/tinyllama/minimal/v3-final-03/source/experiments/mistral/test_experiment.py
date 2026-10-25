"""Small tests for scoring, packed wire representation, and exact transport."""
import unittest
import torch
from experiment import score_qa
from mini_repro.experiment_common import pack_uniform, unpack_uniform
from mini_repro.ttft import rate_limited_transfer, measurement_schedule


class ExperimentTests(unittest.TestCase):
    def test_qa_alias_normalization_and_partial_credit(self):
        self.assertEqual(score_qa('The fruit!', ['fruits', 'Fruit']), {'em':1.0,'f1':1.0})
        self.assertEqual(score_qa('blue flowers', ['red flowers']), {'em':0.0,'f1':0.5})
        self.assertEqual(score_qa('unrelated', ['fruit']), {'em':0.0,'f1':0.0})

    def test_real_packed_size_error_and_zero_groups(self):
        torch.manual_seed(0)
        blob=torch.randn(2,2,3,129,8).half()
        blob[0,0]=0
        p4,p8=pack_uniform(blob,4),pack_uniform(blob,8)
        a,b=unpack_uniform(p4,device='cpu'),unpack_uniform(p8,device='cpu')
        self.assertEqual(a.shape,blob.shape)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.equal(a[0,0],blob[0,0]))
        self.assertLess(len(p4),len(p8)*.7)
        self.assertLess(float((b-blob).abs().mean()),float((a-blob).abs().mean()))

    def test_transport_and_balanced_repeats(self):
        payload=bytes(range(256))*20
        received,seconds=rate_limited_transfer(payload,.1)
        self.assertEqual(received,payload)
        self.assertGreater(seconds,0)
        steps=measurement_schedule(['text','cachegen2'],[.1,1],warmup=1,repeats=3,seed=0)
        self.assertEqual(sum(s['phase']=='measure' for s in steps),12)
        for bw in [.1,1]:
            for method in ['text','cachegen2']:
                self.assertEqual(sum(s['phase']=='measure' and s['bandwidth_gbps']==bw and s['method']==method for s in steps),3)


if __name__=='__main__':unittest.main()
