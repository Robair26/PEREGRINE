"""Lightweight tests, no DIOR data required. Run: PYTHONPATH=src python -m unittest -v test_checkpoint_format.py"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from train_dior_v3 import pack_numpy_rng, unpack_numpy_rng

class CheckpointFormatTests(unittest.TestCase):
    def test_numpy_rng_exact_round_trip(self):
        np.random.seed(42)
        before = np.random.get_state()
        payload = pack_numpy_rng(before)
        np.random.random(25)
        np.random.set_state(unpack_numpy_rng(payload))
        expected_state = np.random.RandomState(42)
        np.testing.assert_array_equal(np.random.random(20), expected_state.random_sample(20))

    def test_weights_only_checkpoint_round_trip(self):
        np.random.seed(42)
        payload = {'schema_version': 2, 'numpy_rng': pack_numpy_rng(np.random.get_state()),
                   'model_state': {'linear.weight': torch.ones(2, 3)},
                   'optimizer_state': {'state': {}, 'param_groups': []},
                   'torch_rng': torch.get_rng_state()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'checkpoint.pth'
            torch.save(payload, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(loaded['schema_version'], 2)
        self.assertTrue(torch.equal(loaded['model_state']['linear.weight'], torch.ones(2, 3)))
        self.assertEqual(len(loaded['numpy_rng'][1]), 624)

if __name__ == '__main__':
    unittest.main()
