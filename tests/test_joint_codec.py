import random
import struct
import unittest

from experiments.joint_codec import (JointCodec, rank_counts, unrank_counts,
    encode_cdf_field, decode_cdf_field, normalize_counts, pack_values, unpack_values,
    fixed_wire_size, inspect_wire, schedule, arithmetic_encode, arithmetic_decode)


class JointCodecTests(unittest.TestCase):
    def test_colex_all_tail_lengths(self):
        rng = random.Random(23)
        for n in range(1, 257):
            for m in (11, 15, 16, 31, 32):
                c = [0] * m
                for _ in range(n):
                    c[rng.randrange(m)] += 1
                self.assertEqual(c, unrank_counts(rank_counts(c), n, m))

    def test_cdf_active_fallback_and_raw(self):
        for q in (1, 2, 3):
            bins = schedule(q, 2)
            b = 16 if q == 1 else 32
            counts = [[2] + [0] * (b-1) for _ in bins]
            for mode in (0x10, 0x11, 0x12):
                field = encode_cdf_field(counts, bins, 2, b, mode)
                self.assertEqual(decode_cdf_field(field, bins, 2, b), [normalize_counts(c) for c in counts])
            counts[0] = [0] * (b-1) + [2]
            self.assertEqual(encode_cdf_field(counts, bins, 2, b)[0], 0x11)
        with self.assertRaises(ValueError): decode_cdf_field(b'\x13', [16],2,16)
        with self.assertRaises(ValueError): decode_cdf_field(b'\x12\xff\xff',[16],2,16)

    def test_bits_examples_and_validation(self):
        self.assertEqual(pack_values([2,7,1,14], 4), bytes.fromhex('72e1'))
        self.assertEqual(pack_values([1,2,3,4,5,6,7,8], 5), bytes.fromhex('410c52cc41'))
        for width in (4,5):
            values = [i % (1<<width) for i in range(128)]
            self.assertEqual(unpack_values(pack_values(values,width),width,len(values)),values)
        with self.assertRaises(ValueError): pack_values([32],5)

    def test_arithmetic_reference(self):
        rng = random.Random(1)
        for n in (1,2,95,96):
            symbols = [rng.randrange(31) for _ in range(n)]
            c = [symbols.count(i) for i in range(32)]
            cdf = normalize_counts(c)
            self.assertEqual(symbols,arithmetic_decode(arithmetic_encode(symbols,cdf),cdf,n))

    def test_three_paths_write_same_final_and_only_selected_branch(self):
        # [L,2,H,N,d], nonzero prefix is deliberately protected.
        source = [[[[[float((t+d)%7-3) for d in range(8)] for t in range(7)]] for _ in range(2)]]
        for q in (1,2,3):
            results = []
            for mode in ('ac_count','ac_cdf','fixed_bits'):
                codec = JointCodec()
                wire = codec.encode(source,q,mode,0,7)
                final = [[[[[99.0]*8 for t in range(7)]] for _ in range(2)]]
                codec.restore(wire,final,0)
                results.append(final)
                meta = inspect_wire(wire)
                self.assertEqual(meta['q'],q)
                if mode == 'fixed_bits':
                    self.assertEqual(len(wire),fixed_wire_size(1,1,8,7,q))
                    self.assertEqual(codec.calls['cdf'],0)
                    self.assertEqual(codec.calls['arithmetic'],0)
                elif mode == 'ac_cdf':
                    self.assertEqual(meta['cdf_tag'],0x10)
                    self.assertEqual(codec.calls['rank'],0)
            self.assertEqual(results[0],results[1])
            self.assertEqual(results[1],results[2])

    def test_all_layer_schedules_and_full_b_diagnostic(self):
        source=[[[[[float((l+t+di)%9-4) for di in range(8)] for t in range(2)]] for s in range(2)] for l in range(32)]
        for q in (1,2,3):
            codec=JointCodec()
            results=[]
            for mode in ('ac_count','ac_cdf','fixed_bits'):
                wire=codec.encode(source,q,mode,0,2)
                final=[[[[[0.0]*8 for t in range(2)]] for s in range(2)] for l in range(32)]
                codec.restore(wire,final,0); results.append(final)
            self.assertEqual(results[0],results[1]); self.assertEqual(results[1],results[2])
            codec.force_full_b=True
            self.assertEqual(inspect_wire(codec.encode(source,q,'ac_count',0,2))['cdf_tag'],0x11)

    def test_arithmetic_segments_and_real_tail_blocks(self):
        for n in (1,95,96,97,255,256):
            source=[[[[[float((t*3+di)%19-9) for di in range(8)] for t in range(n)]] for s in range(2)]]
            for q in (1,2,3):
                results=[]
                for mode in ('ac_count','ac_cdf','fixed_bits'):
                    codec=JointCodec()
                    wire=codec.encode(source,q,mode,256,n+256)
                    final=[[[[[66.0]*8 for t in range(n+256)]] for s in range(2)]]
                    codec.restore(wire,final,256)
                    self.assertEqual(final[0][0][0][0],[66.0]*8)
                    results.append(final)
                self.assertEqual(results[0],results[1]); self.assertEqual(results[1],results[2])

    def test_m45f_lengths_and_corruption(self):
        for q,expected in ((1,134743040),(2,141034496),(3,168297472)):
            self.assertEqual(16*fixed_wire_size(32,8,128,256,q),expected)
        source = [[[[[0.0]*8]] for _ in range(2)]]
        codec = JointCodec()
        wire = codec.encode(source,2,'fixed_bits',0,1)
        v1 = bytearray(wire); struct.pack_into('<H',v1,4,1)
        self.assertEqual(inspect_wire(bytes(v1))['q'],2)
        with self.assertRaises(ValueError): inspect_wire(wire[:-1])

    def test_workspace_uses_capacities_across_requests_and_representations(self):
        from experiments.joint_codec import workspace_requirements,workspace_state_from_snapshot
        snapshot={'encode':{},'restore':{}}
        request=lambda q,m,n,direction: workspace_requirements(q,m,n,32,8,128,direction)
        need=request(2,'fixed_bits',128,'encode')
        self.assertEqual(workspace_state_from_snapshot(snapshot,need,'encode'),'first')
        snapshot['encode'].update(need)
        self.assertEqual(workspace_state_from_snapshot(snapshot,need,'encode'),'reuse')
        self.assertEqual(workspace_state_from_snapshot(snapshot,request(3,'fixed_bits',128,'encode'),'encode'),'expand')
        self.assertEqual(workspace_state_from_snapshot(snapshot,request(2,'fixed_bits',256,'encode'),'encode'),'expand')
        self.assertEqual(workspace_state_from_snapshot(snapshot,request(2,'ac_count',128,'encode'),'encode'),'first')
        self.assertEqual(workspace_state_from_snapshot(snapshot,request(2,'fixed_bits',128,'restore'),'restore'),'first')
        snapshot['encode'].update(request(2,'ac_count',256,'encode'))
        self.assertEqual(workspace_state_from_snapshot(snapshot,request(2,'ac_count',128,'encode'),'encode'),'reuse')
        snapshot={'encode':request(2,'ac_count',96,'encode')}
        self.assertEqual(workspace_state_from_snapshot(snapshot,request(2,'ac_count',97,'encode'),'encode'),'expand')

    def test_cuda_workspace_snapshot_reads_metadata_only(self):
        from experiments.joint_cuda import CudaCodec
        class TensorMetadata:
            def __init__(self,shape): self.shape=shape
            def numel(self):
                result=1
                for x in self.shape: result*=x
                return result
        backend=CudaCodec.__new__(CudaCodec)
        backend._enc_cpu={'fixed_body':TensorMetadata((2048,))}
        backend._dec_cpu={}; backend._dec_gpu={}
        backend._enc_body=TensorMetadata((2048,)); backend._ar_buffer=None
        backend._ar_lengths=None; backend._symbols=None
        codec=JointCodec(device='cuda'); codec._cuda_backend=backend
        snap=codec.workspace_snapshot()
        self.assertEqual(snap,{'encode':{'cpu.fixed_body':2048,'gpu.fixed_body':2048},'restore':{}})
        self.assertEqual(codec.workspace_state(2,'fixed_bits',1,1,1,8,'encode'),'reuse')
        self.assertEqual(codec.workspace_state(2,'ac_count',1,1,1,8,'encode'),'first')

    def test_cuda_encode_failure_synchronizes_before_source_can_be_released(self):
        log=[]
        class Backend:
            def encode_gpu(self,*args):
                log.append('launched'); raise MemoryError('allocation failed')
            def synchronize_safe(self,error=None): log.append('synchronized')
        codec=JointCodec(device='cuda'); codec._cuda_backend=Backend()
        with self.assertRaises(MemoryError): codec.encode_gpu(None,2,'fixed_bits',0,1)
        log.append('source_lease_released')
        self.assertEqual(log,['launched','synchronized','source_lease_released'])

    def test_untrusted_cuda_failure_stops_retry_and_sync_failure_is_fatal(self):
        from experiments.joint_cuda import CudaCodec
        from types import SimpleNamespace
        log=[]
        backend=CudaCodec.__new__(CudaCodec); backend._device_error=False; backend.device='cuda'
        backend.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device:log.append('sync')))
        backend.synchronize_safe(MemoryError('allocation failed'))
        self.assertEqual(log,['sync'])
        with self.assertRaisesRegex(RuntimeError,'device_error'):
            backend.synchronize_safe(RuntimeError('CUDA error: an illegal memory access was encountered'))
        with self.assertRaisesRegex(RuntimeError,'device_error'): backend._check_stream()
        self.assertEqual(log,['sync'])
        backend._device_error=False
        def failed_sync(device): raise RuntimeError('unknown asynchronous CUDA error')
        backend.torch.cuda.synchronize=failed_sync
        with self.assertRaisesRegex(RuntimeError,'device_error'): backend.synchronize_safe()


if __name__ == '__main__': unittest.main()
