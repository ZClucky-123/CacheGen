import random
import unittest
from types import SimpleNamespace
from experiments.joint_scheduler import Snapshot
from experiments.joint_policy import choose_joint, choose_joint_exhaustive, enumerate_plans
from experiments.joint_profiles import RuntimeProfile
from tests.test_joint_scheduler import item


def request(deadline=10**12, fixed=None):
    return SimpleNamespace(clock=SimpleNamespace(deadline_ns=deadline), min_quality=1,
                           quality_mode='fixed' if fixed else 'adaptive', fixed_quality=fixed,
                           total_tokens=1024, validation_only=True, residual_mode='collect')


class PolicyTests(unittest.TestCase):
    def test_highest_feasible_quality_then_fastest_route(self):
        candidates = [item(q=q, encoding=m, g=g) for q in (1, 2, 3)
                      for m, g in [('ac_count', 20), ('ac_cdf', 10), ('fixed_bits', 30)]]
        d = choose_joint(candidates, Snapshot(0), RuntimeProfile(), request())
        self.assertEqual((d.quality, d.encoding), (3, 'ac_cdf'))
        self.assertEqual(d.stats['qualities_evaluated'], [3])
        self.assertEqual(d.stats['plans_evaluated'], 3)

    def test_plan_counts_and_last_block_dedup(self):
        candidates = [item(i, encoding=m) for i in (0, 1)
                      for m in ('ac_count', 'ac_cdf', 'fixed_bits')]
        self.assertEqual(len(enumerate_plans(candidates, 2)), 9)
        self.assertEqual(len(enumerate_plans(candidates[:3], 2)), 3)

    def test_exhaustive_reference_matches_random_small_states(self):
        rng = random.Random(129)
        for _ in range(40):
            candidates = [item(i, q, m, size=rng.randrange(10, 300),
                               g=rng.randrange(1, 100), r=rng.randrange(1, 50),
                               ready=rng.choice((True, False)))
                          for i in (0, 1) for q in (1, 2, 3)
                          for m in ('ac_count', 'ac_cdf', 'fixed_bits')]
            snap = Snapshot(0, history_valid=True, beta50_bps=80e9)
            req = request(rng.randrange(10, 300))
            a = choose_joint(candidates, snap, RuntimeProfile(), req)
            b = choose_joint_exhaustive(candidates, snap, RuntimeProfile(), req)
            self.assertEqual((a.quality, a.encoding, a.plan.signature),
                             (b.quality, b.encoding, b.plan.signature))

    def test_whole_text_has_quality_three_and_only_before_first_send(self):
        req = request()
        req.whole_text = item(q=3, encoding='whole_text', g=0, c=0, r=1)
        profile = RuntimeProfile(text_admitted=True)
        decision = choose_joint([item(q=2)], Snapshot(0), profile, req)
        self.assertEqual(decision.route, 'whole_text')


if __name__ == '__main__':
    unittest.main()
