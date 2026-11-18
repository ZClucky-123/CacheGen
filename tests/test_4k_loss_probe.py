"""Guard diagnostic pairing, isolation, and corruption detection without a GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.diagnose_4k_loss import make_plan, reserve_output, schedules, compare_tensors
import torch


class LossProbePlanTests(unittest.TestCase):
    def fixture(self, root):
        selection = {'quality': [{'source_id': i, 'case_id': f'longchat-{i}-4096',
                                  'context_tokens': 4096, 'context_ids': [1]*4096,
                                  'continuation_ids': list(range(32))} for i in (3,5,16,7)]}
        cfg = {'model': {'name':'mistral', 'layers':32, 'key_boundaries':[10,20,32], 'value_boundary':2},
               'execution': {}, 'cpu_threads':8, 'seed':0}
        records = []
        for case in selection['quality']:
            for method in ('raw','cachegen1','cachegen2','cachegen3'):
                records.append({'kind':'quality', 'source_id':case['source_id'], 'context_tokens':4096,
                                'case_id':case['case_id'], 'method':method, 'scored_ids':list(range(32)),
                                'nll':.5, 'token_nll':[.5]*32})
        contents = {'config.json':json.dumps(cfg), 'selection.json':json.dumps(selection),
                    'metadata.json':json.dumps({'identity':{'inputs':{'code':{}}}}),
                    'records.jsonl':'\n'.join(map(json.dumps,records))+'\n'}
        hashes = {}
        for name, content in contents.items():
            (root/name).write_text(content)
            hashes[name]={'sha256':hashlib.sha256(content.encode()).hexdigest(), 'bytes':len(content.encode())}
        (root/'artifacts.json').write_text(json.dumps({'status':'completed','files':hashes}))

    def test_plan_keeps_original_targets_and_control_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            plan=make_plan(root)
            self.assertEqual([c['source_id'] for c in plan['cases']], [3,5,16,7])
            self.assertEqual(plan['cases'][0]['continuation_ids'],list(range(32)))
            self.assertEqual(plan['score_rows'],44)  # 4 x (raw + 5 variants x 2 paths)

    def test_rejects_corrupt_archive_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            with (root/'selection.json').open('a') as f:f.write(' ')
            with self.assertRaisesRegex(ValueError,'integrity'):
                make_plan(root)

    def test_rejects_changed_target_even_if_file_hash_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            lines=(root/'records.jsonl').read_text().splitlines()
            row=json.loads(lines[1]);row['scored_ids'][0]=999;lines[1]=json.dumps(row)
            data=('\n'.join(lines)+'\n').encode();(root/'records.jsonl').write_bytes(data)
            m=json.loads((root/'artifacts.json').read_text());m['files']['records.jsonl']={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
            (root/'artifacts.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'target'):
                make_plan(root)

    def test_reserves_new_directory_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=root/'batch';batch.mkdir()
            with self.assertRaisesRegex(ValueError,'original batch'):
                reserve_output(batch/'probe',batch)
            output=root/'new';reserve_output(output,batch);(output/'keep').write_text('evidence')
            with self.assertRaises(FileExistsError):reserve_output(output,batch)
            self.assertEqual((output/'keep').read_text(),'evidence')

    def test_single_component_interventions_hold_other_component_fixed(self):
        original=schedules({'layers':32,'key_boundaries':[10,20,32],'value_boundary':2})
        baseline=original['cachegen1']; k=original['cachegen1_k32'];v=original['cachegen1_v32']
        self.assertEqual(k['value_bins'],baseline['value_bins'])
        self.assertEqual(v['key_bins'],baseline['key_bins'])
        self.assertEqual(k['key_bins'],[32]*32)
        self.assertEqual(v['value_bins'],[32]*32)


class LossProbeTensorTests(unittest.TestCase):
    def test_identity_check_catches_single_symbol_and_nonfinite_errors(self):
        import torch
        reference=torch.tensor([1,2,3],dtype=torch.uint8)
        self.assertTrue(compare_tensors(reference,reference.clone())['identical'])
        changed=reference.clone();changed[1]=4
        diff=compare_tensors(reference,changed)
        self.assertFalse(diff['identical']);self.assertEqual(diff['different_elements'],1)
        self.assertFalse(compare_tensors(torch.zeros(2),torch.tensor([0.,float('nan')]))['identical'])


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required: actual arithmetic codec not tested on CPU')
class LossProbeGPUTests(unittest.TestCase):
    def test_real_codec_preserves_quantized_symbols_and_reconstructed_kv(self):
        from types import SimpleNamespace
        from tools.diagnose_4k_loss import make_codecs, quantized_reference, audit_encoded, encode_probe_payload
        spec={'id':'mistral-community/Mistral-7B-v0.2','layers':32,
              'key_boundaries':[10,20,32],'value_boundary':2}
        engine=SimpleNamespace(spec=spec)
        torch.manual_seed(12)
        for length in (257,4096):
            # Small head geometry keeps this a codec test; real model geometry
            # and model NLL are checked only by the actual diagnostic run.
            blob=torch.randn((32,2,2,length,16),device='cuda',dtype=torch.float16)
            blob[:,:,:,0,:]=0
            for name,variant in schedules(spec).items():
                with self.subTest(length=length,variant=name):
                    encoder,decoder=make_codecs(engine,variant)
                    direct,symbols,scales=quantized_reference(blob,encoder)
                    payload,metadata=encode_probe_payload(blob,encoder,symbols,scales)
                    self.assertEqual(metadata['encoder_path'],
                                     'shared_alphabet_diagnostic' if name.startswith('cachegen1_') else 'original_serializer')
                    checks=audit_encoded(payload,symbols,scales)
                    self.assertTrue(all(checks[k]['identical'] for k in ('symbols','key_scales','value_scales')))
                    self.assertEqual(sum(checks['internal_segment_tokens']),length)
                    self.assertTrue(compare_tensors(direct,decoder.from_bytes(payload))['identical'])


if __name__=='__main__':unittest.main()
