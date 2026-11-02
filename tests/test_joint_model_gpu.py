"""Target-device model check; requires the pinned offline Mistral snapshot."""
import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'target CUDA environment required')
class JointModelGPUTests(unittest.TestCase):
    def test_original_codec_and_joint_first_token_for_every_quality(self):
        from experiments.config import load_config
        from experiments.engine import Engine
        from experiments.joint_codec import JointCodec
        from experiments.joint_calibration import require_gpu
        from experiments.joint_config import REPO_ROOT
        import json
        require_gpu()
        cfg=load_config(REPO_ROOT/'experiments/configs/larger-mistral.json')
        engine=Engine(cfg['model'],execution=cfg['execution'])
        row=json.loads((REPO_ROOT/cfg['data']['longchat']).read_text(encoding='utf-8').splitlines()[0])
        ids=engine.tokenizer.encode(row['prompt'],add_special_tokens=False)[:256]
        suffix=engine.tokenizer.encode('\nQuestion: What is the main topic?\nAnswer:',add_special_tokens=False)
        master=engine.prefill(ids)
        original=master.clone()
        for q in (1,2,3):
            old_wire,_,old_restore=engine.encode(master,f'cachegen{q}')
            baseline=old_restore(old_wire)
            baseline_token=engine.first(ids,suffix,baseline)
            for encoding in ('ac_count','ac_cdf','fixed_bits'):
                codec=JointCodec(engine=engine,device='cuda')
                wire=codec.encode(master,q,encoding,0,len(ids))
                final=torch.empty_like(master)
                codec.restore(wire,final,0)
                self.assertTrue(torch.equal(final.view(torch.int16),baseline.contiguous().view(torch.int16)))
                self.assertEqual(engine.first(ids,suffix,final),baseline_token)
                self.assertTrue(torch.equal(master,original))
                self.assertNotEqual(master.data_ptr(),final.data_ptr())
