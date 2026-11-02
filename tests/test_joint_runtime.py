import threading
import time
import unittest
from experiments.joint_runtime import GPUExecutor, JointState
from experiments.joint_types import begin_request, DecisionIdentity, Recipe, PreparedPacket

try:
    import torch
except ImportError:
    torch = None


class RuntimeStateTests(unittest.TestCase):
    def test_stale_decision_cannot_freeze_or_commit(self):
        state = JointState(begin_request(1))
        a = state.select(0, 'r', (3, 'fixed_bits'), state.state_version)
        b = state.select(0, 'r', (2, 'ac_count'), state.state_version)
        recipe = Recipe('ctx', 'r', 0, 0, 1, 3, 'fixed_bits')
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            state.freeze(PreparedPacket(recipe, b'x', a), 10)
        self.assertFalse(state.valid(a))
        self.assertTrue(state.valid(b))
        state.cancel()
        self.assertFalse(state.valid(b))

    def test_started_encoding_is_not_preempted_restore_has_queue_priority(self):
        order = []
        started, finish = threading.Event(), threading.Event()
        with GPUExecutor() as gpu:
            first = gpu.submit('encode', lambda: (started.set(), finish.wait(1), order.append('first')))
            self.assertTrue(started.wait(1))
            second = gpu.submit('encode', lambda: order.append('second'))
            restore = gpu.submit('restore', lambda: order.append('restore'))
            finish.set()
            first.result(2)
            second.result(2)
            restore.result(2)
        self.assertEqual(order, ['first', 'restore', 'second'])

    @unittest.skipIf(torch is None, 'CPU torch required for tensor integration')
    def test_multiblock_virtual_codec_real_socket_tail_and_pipeline_snapshots(self):
        """Large fake wires trigger causal feedback while tiny KV keeps CPU QA small."""
        from experiments.joint_runtime import Runtime
        from experiments.joint_profiles import RuntimeProfile
        from experiments.joint_wire import frame, unframe
        class Engine:
            spec = {'layers': 2, 'kv_heads': 1, 'head_dim': 8}
            def allocate_final(self, total):
                return torch.empty(2, 2, 1, total, 8, dtype=torch.float16)
            def first(self, context, suffix, final):
                return 42
        class Codec:
            def __init__(self):
                self.device = 'cpu'
                self.pieces = {}
                self.generated = []
            def encode_gpu(self, piece, quality, encoding, start, total):
                self.generated.append((start, quality, encoding))
                self.pieces[start] = piece.clone()
                time.sleep(.004)
                return start
            def seal(self, start):
                return frame(start.to_bytes(4, 'little') + bytes(262144),
                             {'token_start': start, 'token_end': start+self.pieces[start].shape[3],
                              'method': 'fixed_bits'})
            def restore(self, wire, final, start):
                _, body = unframe(wire)
                assert int.from_bytes(body[:4], 'little') == start
                piece = self.pieces[start]
                time.sleep(.002)
                final[:, :, :, start:start+piece.shape[3], :] = piece
        raw = torch.arange(2*2*513*8).reshape(2,2,1,513,8).half()
        for mode in ('serial', 'pipeline'):
            costs = [dict(q=2, encoding='fixed_bits', N=min(256, 513-start), prefix=start,
                          mode=mode, g50_s=.004, g90_s=.006, c50_s=.0001, c90_s=.0002,
                          r50_s=.002, r90_s=.004, size50=262148, size90=262148)
                     for start in (0, 256, 512)]
            profile = RuntimeProfile(synthetic=True, costs=costs)
            codec = Codec()
            runtime = Runtime(Engine(), codec, profile, {'qualities': [2],
                              'encodings': ['fixed_bits'], 'execution_mode': mode})
            ctx = runtime.register_context(list(range(513)), raw, 'original')
            try:
                outcome = runtime.run(ctx, [1], 2, bandwidth_bps=1e9, quality_mode='fixed',
                                      fixed_quality=2, validation_only=True, residual_mode='collect')
                self.assertEqual(codec.generated, [(i, 2, 'fixed_bits') for i in (0, 256, 512)])
                self.assertTrue(torch.equal(outcome['final_kv'], raw))
                self.assertEqual(len(outcome['network']), 3)
                restored = [e['identity']['block_index'] for e in outcome['events'] if e['event']=='RESTORE_DONE']
                self.assertEqual(restored, [0, 1, 2])
                self.assertEqual(outcome['token'], 42)
            finally:
                runtime.close_context(ctx)

    @unittest.skipIf(torch is None, 'CPU torch required for tensor integration')
    def test_real_three_path_requests_cache_reuse_and_master_release(self):
        from experiments.joint_runtime import Runtime
        from experiments.joint_profiles import RuntimeProfile
        from experiments.joint_codec import JointCodec
        class Engine:
            spec = {'layers': 2, 'kv_heads': 1, 'head_dim': 8}
            def allocate_final(self, total):
                return torch.empty(2, 2, 1, total, 8, dtype=torch.float16)
            def first(self, context, suffix, final):
                self.final = final
                return 42
        raw = torch.linspace(-2, 2, 2*2*8*8).reshape(2,2,1,8,8).half()
        profile = RuntimeProfile(synthetic=True,costs=[dict(q=2,encoding=m,N=8,prefix=0,
            mode='pipeline',g50_s=.001,g90_s=.002,c50_s=.001,c90_s=.002,
            r50_s=.001,r90_s=.002,size50=2000,size90=3000) for m in ('ac_count','ac_cdf','fixed_bits')])
        outputs=[]
        for method in ('ac_count','ac_cdf','fixed_bits'):
            codec=JointCodec(device='cpu')
            runtime=Runtime(Engine(),codec,profile,{'qualities':[2],'encodings':[method]})
            ctx=runtime.register_context(list(range(8)),raw,'original')
            first=runtime.run(ctx,[1],1,quality_mode='fixed',fixed_quality=2,
                              validation_only=True,residual_mode='collect')
            calls=codec.calls['quantize']
            runtime.sources.release_master(ctx.context_id)
            second=runtime.run(ctx,[1],1,quality_mode='fixed',fixed_quality=2,
                               validation_only=True,residual_mode='collect')
            self.assertEqual(codec.calls['quantize'],calls)
            self.assertEqual(second['generation'],[])
            self.assertEqual(first['token'],42)
            self.assertIsNot(first['final_kv'],second['final_kv'])
            self.assertTrue(torch.equal(first['final_kv'],second['final_kv']))
            self.assertEqual(len(first['generation']),1)
            outputs.append(first['final_kv'])
            runtime.close_context(ctx)
        self.assertTrue(all(torch.equal(outputs[0],other) for other in outputs[1:]))

    @unittest.skipIf(torch is None, 'CPU torch required for tensor integration')
    def test_text_recomputes_with_actual_prefix_and_whole_is_one_packet(self):
        import json
        from experiments.joint_runtime import Runtime
        from experiments.joint_profiles import RuntimeProfile
        from experiments.joint_codec import JointCodec
        class Engine:
            spec={'layers':1,'kv_heads':1,'head_dim':8}
            def __init__(self): self.prefixes=[]
            def allocate_final(self,total): return torch.empty(1,2,1,total,8,dtype=torch.float16)
            def text_payload(self,ids): return json.dumps(ids).encode()
            def restore_text(self,wire): return json.loads(wire)
            def prefill(self,ids,prefix=None):
                self.prefixes.append(0 if prefix is None else prefix.shape[3])
                tail=torch.ones(1,2,1,len(ids),8,dtype=torch.float16)
                if prefix is None: return tail
                assert torch.all(prefix==1)
                return torch.cat((prefix,tail),dim=3)
            def first(self,context,suffix,final): return 7
        costs=[dict(q=3,encoding=m,N=n,prefix=p,mode='pipeline',g50_s=0,g90_s=0,
            c50_s=.001,c90_s=.002,r50_s=.001,r90_s=.002,size50=100,size90=200)
            for m,n,p in (('text',256,0),('text',4,256),('whole_text',260,0))]
        for route,prefixes,packets in (('chunks',[0,256],2),('whole_text',[0],1)):
            engine=Engine()
            runtime=Runtime(engine,JointCodec(device='cpu'),RuntimeProfile(synthetic=True,
                text_admitted=True,costs=costs),{'qualities':[3]})
            ctx=runtime.register_context(list(range(260)),None,'r')
            result=runtime.run(ctx,[1],1,validation_only=True,residual_mode='collect',route=route)
            self.assertEqual(engine.prefixes,prefixes)
            self.assertEqual(result['route'],route)
            self.assertEqual(len(result['network']),packets)
            self.assertTrue(torch.all(result['final_kv']==1))
            runtime.close_context(ctx)
