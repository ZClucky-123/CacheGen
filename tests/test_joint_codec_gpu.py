"""Run after building both CUDA extensions; CPU skips are not GPU passes."""
import unittest
try:
    import torch
except ImportError:
    torch=None

from experiments.joint_codec import (JointCodec,schedule,inspect_wire,
    decode_cdf_field,rank_bytes)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'target CUDA environment required')
class JointCodecGPUTests(unittest.TestCase):
    def test_three_qualities_paths_tail_symbols_and_final_bits(self):
        from lmcache.storage_backend.serde.cachegen_encoder import torch_quant_vectorized
        from lmcache.storage_backend.serde.cachegen_decoder import do_dequantize
        torch.manual_seed(17)
        codecs={mode:JointCodec(device='cuda') for mode in ('ac_count','ac_cdf','fixed_bits')}
        codecs['ac_count'].preinitialize_combinadic_table()
        # All layer boundaries, a noncontiguous token slice, and protected prefix.
        for n in range(1,257):
            master=torch.randn(32,2,1,n+256,8,device='cuda',dtype=torch.float16)
            original=master.clone()
            source=master[:,:,:,256:,:]
            for q in (1,2,3):
                bins=torch.tensor(schedule(q,32),device='cuda',dtype=torch.float32)
                planes=source.permute(1,0,3,2,4).reshape(64,n,8)
                symbols,scales=torch_quant_vectorized(bins,planes)
                expected=do_dequantize(symbols.float(),bins,scales).half().reshape(2,32,n,1,8).permute(1,0,3,2,4)
                for mode in ('ac_count','ac_cdf','fixed_bits'):
                    codec=codecs[mode]
                    wire=codec.encode(source,q,mode,256,n+256)
                    final=torch.full((32,2,1,n+256,8),77,dtype=torch.float16,device='cuda')
                    codec.restore(wire,final,256)
                    self.assertTrue(torch.equal(final[:,:,:,256:,:].view(torch.int16),expected.contiguous().view(torch.int16)))
                    self.assertTrue(bool((final[:,:,:,:256,:]==77).all()))
                    self.assertTrue(torch.equal(master,original))
                    if mode=='fixed_bits':
                        self.assertEqual(codec.calls['cdf'],0); self.assertEqual(codec.calls['arithmetic'],0)
                    elif mode=='ac_cdf':
                        self.assertEqual(codec.calls['rank'],0)
                        self.assertIsNone(codec._cuda_backend._table)

    def test_all_tail_lengths_exact_cdf_and_full_b_fallback(self):
        from experiments.joint_cuda import CudaCodec
        from collections import Counter
        import torchac_cuda
        backend=CudaCodec(Counter(),'cuda')
        ext=backend.ext
        for q in (1,2,3):
            bins=schedule(q,32); b=16 if q==1 else 32
            for n in range(1,257):
                symbols=torch.empty((64,n,8),device='cuda',dtype=torch.int8)
                for p,bn in enumerate(bins): symbols[p]=torch.randint(0,bn-1,(n,8),device='cuda',dtype=torch.int8)
                table,sizes=backend._combinadic(n)
                cdf,field,error=ext.calculate_cdf_and_rank(symbols,q,False,table,sizes)
                self.assertEqual(int(error.item()),0)
                baseline=torchac_cuda.calculate_cdf(symbols,b)
                self.assertTrue(torch.equal(cdf,baseline))
                restored,err=ext.restore_cdf(field,n,q,32,8,table,sizes)
                self.assertEqual(int(err.item()),0)
                self.assertTrue(torch.equal(restored,baseline))
                if n in (1,256):
                    blob=field.cpu().numpy().tobytes()
                    reference=decode_cdf_field(blob,[bn for bn in bins for _ in range(8)],n,b)
                    unsigned=cdf.to(torch.int32).bitwise_and(65535).reshape(-1,b+1).cpu().tolist()
                    self.assertEqual(reference,unsigned)
            symbols.fill_(b-1)
            cdf,field,error=ext.calculate_cdf_and_rank(symbols,q,False,table,sizes)
            self.assertEqual(int(error.item()),1)
            cdf,field,error=ext.calculate_cdf_and_rank(symbols,q,True,table,sizes)
            self.assertEqual(int(error.item()),0)
            restored,err=ext.restore_cdf(field,256,q,32,8,table,sizes)
            self.assertEqual(int(err.item()),0); self.assertTrue(torch.equal(cdf,restored))
            corrupt=field.clone(); corrupt[1]=255
            _,err=ext.restore_cdf(corrupt,256,q,32,8,table,sizes)
            self.assertNotEqual(int(err.item()),0)

    def test_pinned_slots_lease_force_full_and_default_stream(self):
        source=torch.zeros((1,2,1,1,8),dtype=torch.float16,device='cuda')
        codec=JointCodec(device='cuda'); codec.force_full_b=True
        staged=codec.encode_gpu(source,2,'ac_count',0,1)
        with self.assertRaises(RuntimeError): codec.encode_gpu(source,2,'fixed_bits',0,1)
        wire=codec.seal(staged)
        self.assertEqual(inspect_wire(wire)['cdf_tag'],0x11)
        codec.reset_buffers()
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaises(RuntimeError): codec.encode(source,2,'fixed_bits',0,1)


if __name__=='__main__': unittest.main()
