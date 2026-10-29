"""Regression checks for stage boundaries; standard library only, no GPU imports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper_compression.provenance import provenance, source_paths, ROOT
from tools.paper_stages import manifest


class ProvenanceTests(unittest.TestCase):
    def test_scope_contains_shared_code_and_excludes_network_and_tests(self):
        core, network = provenance('stage1'), provenance('stage2')
        self.assertIn('paper_compression/quantization.py', core)
        self.assertIn('tools/paper_study.py', core)
        for name in ('paper_compression/network.py', 'paper_compression/policy.py',
                     'experiments/ttft.py', 'mini_repro/transport.py', 'tools/run_paper_core.sh'):
            self.assertNotIn(name, core)
            self.assertIn(name, network)
        self.assertFalse(any(p.startswith('tests/') for p in core))

    def test_network_edits_do_not_invalidate_stage1_but_codec_edits_do(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in source_paths('stage2'):
                target = root / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            before = provenance('stage1', root)
            network_before = provenance('stage2', root)
            for name in ('network', 'policy'):
                path = root / f'paper_compression/{name}.py'
                path.write_text(path.read_text() + '\n# network-only correction\n')
            self.assertEqual(before, provenance('stage1', root))
            self.assertNotEqual(network_before, provenance('stage2', root))
            path = root / 'paper_compression/quantization.py'
            path.write_text(path.read_text() + '\n# compression correction\n')
            self.assertNotEqual(before, provenance('stage1', root))

    def test_seal_preserves_artifact_integrity_and_stage1_source_checks(self):
        sources = {'core.py': 'original'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase in ('insights', 'profiles', 'validation', 'test'):
                folder = root / phase
                folder.mkdir()
                (folder / 'status.json').write_text(json.dumps({'status': 'completed'}))
                (folder / 'source_hashes.json').write_text(json.dumps(sources))
            for phase in ('validation-plots', 'test-plots'):
                folder = root / phase
                folder.mkdir()
                for length in (1024, 2048, 4096):
                    for suffix in ('png', 'pdf'):
                        (folder / f'curve-{length}.{suffix}').write_bytes(b'plot')
            (root / 'test-plots/encoding-error.pdf').write_bytes(b'plot')
            with patch('tools.paper_stages.provenance', return_value=sources):
                before = manifest(root)
                (root / 'test/records.jsonl').write_text('changed\n')
                self.assertNotEqual(before, manifest(root))
            with patch('tools.paper_stages.provenance', return_value={'core.py': 'changed'}):
                with self.assertRaises(ValueError):
                    manifest(root)


if __name__ == '__main__':
    unittest.main()
