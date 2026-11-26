"""Server-only checks for the full-length transport experiment wrapper."""
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.full_length_pipeline import ROOT, matrix, replay


class FullLengthPipelineTests(unittest.TestCase):
    def test_matrix_pairs_modes_bandwidths_and_three_repeats(self):
        cfg=json.loads((ROOT/'paper_compression/configs/full-length-pipeline.json').read_text())
        keys=matrix(cfg,cfg['source_ids'])
        self.assertEqual(len(keys),1680)
        self.assertEqual(len(set(keys)),1680)
        self.assertEqual(len(matrix(cfg,[0])),84)
        self.assertEqual({k[2] for k in keys if k[1]=='text'},{'whole'})

    def test_replay_observes_actual_overlap_and_preserves_bytes(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA server required')
        cfg={'queue_capacity':1,'pacing_quantum_bytes':65536}
        packets=[SimpleNamespace(wire=bytes([i])*2048,end=i+1) for i in range(4)]
        engine=SimpleNamespace(first_state=lambda *args:(7,None))
        for mode in ['chunk_serial','chunk_pipeline']:
            def restore(engine,packet,wire,merged):
                self.assertEqual(wire,packet.wire)
                time.sleep(.02)
                return torch.zeros((1,2,1,packet.end,1))
            with patch('experiments.chunks.restore_chunk',side_effect=restore):
                blob,initial,row=replay(engine,cfg,packets,mode,None,[1])
            self.assertEqual(blob.shape[3],4)
            self.assertEqual(initial[0],7)
            self.assertEqual(row['bytes'],8192)
            self.assertTrue(row['byte_identity_checked'])
            if mode=='chunk_serial':self.assertEqual(row['overlap_seconds'],0.)
            else:self.assertGreater(row['overlap_seconds'],0.)
            self.assertGreaterEqual(row['ttft_seconds'],row['restore_sum_seconds'])


if __name__=='__main__':
    unittest.main()
