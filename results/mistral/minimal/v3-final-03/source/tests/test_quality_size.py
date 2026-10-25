import unittest

import torch

from mini_repro.quality_size import shifted_nll


class QualitySizeTests(unittest.TestCase):
    def test_nll_scores_next_tokens_and_excludes_first_suffix_token(self):
        # The first supplied suffix token has no corresponding logit from the
        # restored context cache. Positions 0 and 1 predict suffix tokens 1, 2.
        logits = torch.tensor([[[0., 8., 0.], [0., 0., 8.], [8., 0., 0.]]])
        ids = [0, 1, 2]
        loss = shifted_nll(logits, ids)
        self.assertLess(loss, 0.001)
        self.assertGreater(shifted_nll(logits, [2, 2, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
