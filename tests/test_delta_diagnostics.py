"""Server-only regression checks for matching-position variance measurements."""
import unittest
import torch
from tools.delta_diagnostics import layer_stats,measures


class DeltaStatisticsTests(unittest.TestCase):
    def test_nonanchor_positions_and_direct_variance_reference(self):
        v=torch.tensor([[i+100.,3*i-200.] for i in range(30)])
        result=layer_stats(v,10)
        idx=torch.tensor([i for i in range(30) if i%10])
        delta=v[idx]-v[(idx//10)*10]
        self.assertEqual(result['raw']['tokens'],27)
        self.assertEqual(result['raw']['per_channel_variance'],v[idx].double().var(0,unbiased=False).tolist())
        self.assertEqual(result['delta_original']['per_channel_variance'],delta.double().var(0,unbiased=False).tolist())

    def test_channel_offsets_do_not_fake_token_variance(self):
        v=torch.tensor([[i,2*i] for i in range(10)],dtype=torch.float32)
        a,_=measures(v);b,_=measures(v+torch.tensor([1000.,-1000.]))
        self.assertEqual(a['mean_channel_variance'],b['mean_channel_variance'])
        self.assertNotEqual(a['global_variance'],b['global_variance'])

    def test_zero_variance_channels_are_explicit(self):
        result=layer_stats(torch.ones((20,3)),10)
        self.assertEqual(result['raw']['valid_ratio_channels'],0)
        self.assertIsNone(result['delta_original']['ratio_of_mean_channel_variances'])


if __name__=='__main__':
    unittest.main()
