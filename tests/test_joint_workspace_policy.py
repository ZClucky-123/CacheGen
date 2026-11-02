"""Workspace state comes from actual byte capacities across requests/routes."""
import unittest
from dataclasses import replace
from types import SimpleNamespace

from experiments.joint_codec import workspace_requirements
from experiments.joint_profiles import RuntimeProfile
from experiments.joint_runtime import Runtime, JointState
from experiments.joint_scheduler import CandidateEstimate, Plan, Snapshot, simulate_plan
from experiments.joint_types import begin_request


def requirements(q, encoding, direction):
    return tuple(workspace_requirements(q, encoding, 256, 32, 8, 128, direction).items())


def candidate(index, q=2, encoding='fixed_bits'):
    return CandidateEstimate(index,index*256,(index+1)*256,q,encoding,(index,q,encoding),
                             100,100,999,999,999,999,999,999,
                             encode_requirements=requirements(q,encoding,'encode'),
                             restore_requirements=requirements(q,encoding,'restore'))


def costs(indices=(0,1)):
    rows = []
    for index in indices:
        for q in (2,3):
            for encoding in ('fixed_bits','ac_cdf'):
                for state, seconds in (('first',100e-9),('expand',20e-9),('reuse',1e-9)):
                    rows.append(dict(q=q,encoding=encoding,N=256,prefix=index*256,
                                     buffer_state=state,mode='serial',g50_s=seconds,g90_s=seconds,
                                     c50_s=seconds,c90_s=seconds,r50_s=seconds,r90_s=seconds,
                                     size50=100,size90=100))
    return rows


class WorkspacePolicyTests(unittest.TestCase):
    def test_future_q3_expands_after_first_q2_then_cross_request_reuses(self):
        first, second = candidate(0), candidate(1, q=3)
        profile = RuntimeProfile(costs=costs())
        snap = Snapshot(0,mode='serial',beta50_bps=80e9,workspace={'encode':{},'restore':{}})
        result = simulate_plan(Plan(3,(first,second)),snap,profile)
        self.assertEqual(result.gpu_encode_ns,120)
        self.assertEqual(result.gpu_restore_ns,120)
        self.assertEqual(result.workspace['encode'],dict(second.encode_requirements))
        self.assertEqual(snap.workspace,{'encode':{},'restore':{}})
        warm = simulate_plan(Plan(3,(replace(second,block_index=0,start=0,end=256),)),
                             replace(snap,workspace=result.workspace),profile)
        self.assertEqual(warm.gpu_encode_ns,1)
        self.assertEqual(warm.gpu_restore_ns,1)

    def test_first_arithmetic_representation_is_first_even_after_fixed_bits(self):
        first, second = candidate(0), candidate(1,encoding='ac_cdf')
        result = simulate_plan(Plan(2,(first,second)),
                               Snapshot(0,mode='serial',beta50_bps=80e9),RuntimeProfile(costs=costs()))
        self.assertEqual(result.gpu_encode_ns,200)
        self.assertEqual(result.gpu_restore_ns,200)
        self.assertIn('gpu.fixed_body',result.workspace['encode'])
        self.assertTrue(any(k.startswith('gpu.ar_buffer') for k in result.workspace['encode']))

    def test_runtime_candidates_use_actual_cross_request_capacities(self):
        class Codec:
            workspace = {'encode':{},'restore':{}}
            def workspace_snapshot(self):
                return {d:dict(v) for d,v in self.workspace.items()}
        codec = Codec()
        engine = SimpleNamespace(spec={'layers':32,'kv_heads':8,'head_dim':128})
        profile = RuntimeProfile(costs=costs((0,)))
        runtime = Runtime(engine,codec,profile,{'qualities':[2,3]})
        runtime._mode='serial'
        ctx = runtime.register_context(list(range(256)),object(),'original')
        state = JointState(begin_request(1))
        try:
            codec.workspace={d:dict(requirements(2,'fixed_bits',d)) for d in ('encode','restore')}
            candidates = runtime._candidates(ctx,0,{},state,('fixed_bits','ac_cdf'))
            by_key={(c.q,c.encoding):c for c in candidates}
            self.assertEqual(by_key[2,'fixed_bits'].g50_ns,1)
            self.assertEqual(by_key[3,'fixed_bits'].g50_ns,20)
            self.assertEqual(by_key[2,'ac_cdf'].g50_ns,100)
            self.assertEqual(by_key[2,'fixed_bits'].encode_buffer_state,'reuse')
            self.assertEqual(by_key[3,'fixed_bits'].restore_buffer_state,'expand')
        finally:
            runtime.close_context(ctx)


if __name__ == '__main__':
    unittest.main()
