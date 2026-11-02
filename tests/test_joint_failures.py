"""CPU integration checks for failure, cancellation and late-result lifetimes."""
import threading
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from experiments.joint_codec import JointCodec
from experiments.joint_profiles import RuntimeProfile
from experiments.joint_runtime import Runtime
from mini_repro.transport import DynamicTransport


class FieldLease:
    def __init__(self):
        self.releases = 0

    def release_fields(self):
        self.releases += 1


class ControlledCodec(JointCodec):
    def __init__(self, *, block=None, fail=None):
        super().__init__(device='cpu')
        self.block, self.fail = block, fail
        self.entered = threading.Event()
        self.release = threading.Event()
        self.attempts, self.fields = [], []

    def _wait(self, stage):
        if self.block == stage:
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError('Test worker release timed out')

    def encode_gpu(self, source, q, encoding, start, total_tokens):
        self.attempts.append((q, encoding))
        if self.fail == 'encode' and encoding == 'ac_count':
            raise MemoryError('injected allocation failure')
        fields = super().encode_gpu(source, q, encoding, start, total_tokens)
        fields.owner = FieldLease()
        self.fields.append(fields)
        self._wait('encode')
        return fields

    def _seal(self, fields):
        self._wait('seal')
        if self.fail == 'seal' and fields.metadata['encoding'] == 'ac_count':
            raise MemoryError('injected CPU sealing failure')
        return super()._seal(fields)

    def restore(self, wire, final, start):
        self._wait('restore')
        return super().restore(wire, final, start)


class TinyEngine:
    spec = {'layers': 2, 'kv_heads': 1, 'head_dim': 8}

    def __init__(self):
        self.first_calls = 0

    def allocate_final(self, total):
        return torch.empty(2, 2, 1, total, 8, dtype=torch.float16)

    def first(self, context, suffix, final):
        self.first_calls += 1
        return 42


@unittest.skipIf(torch is None, 'CPU torch required for failure integration')
class JointFailureTests(unittest.TestCase):
    def make_runtime(self, codec, encodings=('ac_count', 'ac_cdf')):
        rows = [dict(q=2, encoding=m, N=8, prefix=0, mode='pipeline',
                     g50_s=.0001 if m == 'ac_count' else .01,
                     g90_s=.0002 if m == 'ac_count' else .02,
                     c50_s=.0001, c90_s=.0002, r50_s=.0001, r90_s=.0002,
                     size50=64, size90=128) for m in encodings]
        runtime = Runtime(TinyEngine(), codec, RuntimeProfile(synthetic=True, costs=rows),
                          {'qualities': [2], 'encodings': list(encodings)})
        raw = torch.ones(2, 2, 1, 8, 8, dtype=torch.float16)
        context = runtime.register_context(list(range(8)), raw, 'original')
        return runtime, context

    def run_request(self, runtime, context):
        return runtime.run(context, [1], 1, quality_mode='fixed', fixed_quality=2,
                           validation_only=True, residual_mode='collect')

    def start_request(self, runtime, context):
        outcome = {}

        def execute():
            try:
                outcome['result'] = self.run_request(runtime, context)
            except BaseException as error:
                outcome['error'] = error

        worker = threading.Thread(target=execute, name='test-request', daemon=True)
        worker.start()
        return worker, outcome

    def assert_clean(self, runtime, context, codec):
        self.assertEqual(context.readers, 0)
        cache = runtime.caches[context.context_id]
        self.assertFalse(cache.pending)
        self.assertTrue(all(entry.pins == 0 for entry in cache.entries.values()))
        self.assertTrue(all(fields.owner.releases == 1 for fields in codec.fields))
        self.assertIsNone(runtime.current_state)
        self.assertIsNone(runtime.current_transport)

    def test_cancel_inflight_encode_and_seal_releases_only_after_completion(self):
        for stage in ('encode', 'seal'):
            with self.subTest(stage=stage):
                codec = ControlledCodec(block=stage)
                runtime, context = self.make_runtime(codec)
                worker, outcome = self.start_request(runtime, context)
                try:
                    self.assertTrue(codec.entered.wait(3))
                    state = runtime.current_state
                    runtime.cancel()
                    self.assertFalse(state.active)
                    self.assertEqual(codec.fields[0].owner.releases, 0)
                    if stage == 'encode':
                        self.assertEqual(context.readers, 1)
                    codec.release.set()
                    worker.join(3)
                    self.assertFalse(worker.is_alive())
                    self.assertNotIn('result', outcome)
                    self.assertRegex(str(outcome['error']), 'request_cancelled')
                    self.assertEqual(state.committed_tokens, 0)
                    self.assertEqual(runtime.engine.first_calls, 0)
                    self.assert_clean(runtime, context, codec)
                finally:
                    codec.release.set()
                    worker.join(5)
                    runtime.close_context(context)

    def test_recoverable_encode_or_seal_failure_materializes_only_fallback(self):
        for stage in ('encode', 'seal'):
            with self.subTest(stage=stage):
                codec = ControlledCodec(fail=stage)
                runtime, context = self.make_runtime(codec)
                try:
                    result = self.run_request(runtime, context)
                    self.assertEqual(codec.attempts, [(2, 'ac_count'), (2, 'ac_cdf')])
                    self.assertEqual([g['encoding'] for g in result['generation'] if g.get('sent')], ['ac_cdf'])
                    self.assertEqual(sum(g.get('failed', False) for g in result['generation']), 1)
                    self.assertEqual(len(result['network']), 1)
                    self.assertTrue(torch.all(result['final_kv'] == 1))
                    self.assert_clean(runtime, context, codec)
                finally:
                    runtime.close_context(context)

    def test_cancel_inflight_restore_does_not_commit_or_return_final(self):
        codec = ControlledCodec(block='restore')
        runtime, context = self.make_runtime(codec)
        worker, outcome = self.start_request(runtime, context)
        try:
            self.assertTrue(codec.entered.wait(3))
            state = runtime.current_state
            runtime.cancel()
            codec.release.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertNotIn('result', outcome)
            self.assertIn('error', outcome)
            self.assertEqual(state.committed_tokens, 0)
            self.assertEqual(runtime.engine.first_calls, 0)
            self.assert_clean(runtime, context, codec)
        finally:
            codec.release.set()
            worker.join(5)
            runtime.close_context(context)

    def test_cancel_inflight_first_token_does_not_return_success(self):
        class BlockingFirst(TinyEngine):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def first(self, context, suffix, final):
                self.entered.set()
                if not self.release.wait(5):
                    raise RuntimeError('Test first-token release timed out')
                return super().first(context, suffix, final)

        codec = ControlledCodec()
        runtime, context = self.make_runtime(codec)
        runtime.engine = BlockingFirst()
        worker, outcome = self.start_request(runtime, context)
        try:
            self.assertTrue(runtime.engine.entered.wait(3))
            state = runtime.current_state
            runtime.cancel()
            runtime.engine.release.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertNotIn('result', outcome)
            self.assertRegex(str(outcome['error']), 'request_cancelled')
            self.assertFalse(state.active)
            self.assert_clean(runtime, context, codec)
        finally:
            runtime.engine.release.set()
            worker.join(5)
            runtime.close_context(context)

    def test_late_result_requires_new_identity_before_send_without_reencoding(self):
        codec = ControlledCodec(block='encode')
        runtime, context = self.make_runtime(codec)
        worker, outcome = self.start_request(runtime, context)
        try:
            self.assertTrue(codec.entered.wait(3))
            state = runtime.current_state
            old = state.choices[0]
            state.select(0, context.source_revision, (2, 'ac_cdf'), state.state_version)
            codec.release.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertNotIn('error', outcome)
            result = outcome['result']
            self.assertEqual(codec.attempts, [(2, 'ac_count')])
            self.assertEqual(len(result['network']), 1)
            self.assertGreater(result['network'][0]['identity']['decision_id'], old.decision_id)
            self.assert_clean(runtime, context, codec)
        finally:
            codec.release.set()
            worker.join(5)
            runtime.close_context(context)

    def test_disconnected_socket_aborts_request_and_releases_workers(self):
        import socket
        links = []

        class DisconnectTransport(DynamicTransport):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                links.append(self)

            def _transfer(self, wire, identity, on_start):
                def disconnect(now):
                    start = on_start(now)
                    self.receiver.shutdown(socket.SHUT_RDWR)
                    return start
                return super()._transfer(wire, identity, disconnect)

        codec = ControlledCodec()
        runtime, context = self.make_runtime(codec)
        try:
            with patch('experiments.joint_runtime.DynamicTransport', DisconnectTransport):
                with self.assertRaises((RuntimeError, OSError)) as caught:
                    self.run_request(runtime, context)
            self.assertIn('joint_failure', vars(caught.exception))
            self.assertEqual(runtime.engine.first_calls, 0)
            self.assertFalse(any(e['event'] == 'RESTORE_DONE' for e in runtime.last_failure['events']))
            self.assertTrue(links[0].cancelled.is_set())
            self.assertEqual(links[0].sender.fileno(), -1)
            self.assertEqual(links[0].receiver.fileno(), -1)
            self.assertTrue(all(not worker.is_alive() for worker in links[0].pool._threads))
            self.assert_clean(runtime, context, codec)
        finally:
            runtime.close_context(context)
