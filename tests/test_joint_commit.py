"""Commit-time expiry must trigger selection again before materialization."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'CPU torch required for runtime commit integration')
class CommitTests(unittest.TestCase):
    def test_expired_high_quality_prediction_reselects_before_encoding(self):
        from experiments.joint_runtime import Runtime
        from experiments.joint_profiles import RuntimeProfile
        from experiments.joint_codec import JointCodec
        from experiments.joint_policy import choose_joint,refresh_decision
        class Engine:
            spec={'layers':1,'kv_heads':1,'head_dim':8}
            def allocate_final(self,total): return torch.empty(1,2,1,total,8,dtype=torch.float16)
            def first(self,context,suffix,final): return 1
        costs=[dict(q=q,encoding='fixed_bits',N=8,prefix=0,mode='pipeline',
            g50_s=.001,g90_s=.002,c50_s=0,c90_s=0,r50_s=.001,r90_s=.002) for q in (2,3)]
        runtime=Runtime(Engine(),JointCodec(device='cpu'),RuntimeProfile(synthetic=True,costs=costs),
                        {'qualities':[2,3],'encodings':['fixed_bits']})
        ctx=runtime.register_context(list(range(8)),torch.ones(1,2,1,8,8,dtype=torch.float16),'r')
        calls=[]
        def choose(candidates,snapshot,profile,request):
            calls.append(snapshot.now_ns)
            if len(calls)>1:
                request=SimpleNamespace(**vars(request))
                request.quality_mode='fixed'
                request.fixed_quality=2
            return choose_joint(candidates,snapshot,profile,request)
        def refresh(decision,*args):
            refreshed=refresh_decision(decision,*args)
            return replace(refreshed,predicted_slo_miss=True) if len(calls)==1 else refreshed
        try:
            with patch('experiments.joint_policy.choose_joint',choose),patch('experiments.joint_policy.refresh_decision',refresh):
                result=runtime.run(ctx,[1],1,validation_only=True,residual_mode='collect')
            self.assertGreaterEqual(len(calls),2)
            self.assertEqual([g['q'] for g in result['generation']],[2])
            self.assertEqual(runtime.codec.calls['quantize'],1)
        finally:
            runtime.close_context(ctx)
