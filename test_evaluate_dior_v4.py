import unittest
from evaluate_dior_v4 import calculate_metrics

class EvaluationMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        x = calculate_metrics([0, 1, 1], [0, 1, 1], ['A', 'B'])
        self.assertEqual(x['accuracy'], 1.0)
        self.assertEqual(x['macro_f1'], 1.0)
        self.assertEqual(x['confusion_matrix'], [[1, 0], [0, 2]])

    def test_missing_predictions_are_zero_division_safe(self):
        x = calculate_metrics([0, 1], [0, 0], ['A', 'B'])
        self.assertEqual(x['accuracy'], 0.5)
        self.assertEqual(x['per_class'][1]['recall'], 0.0)
        self.assertEqual(x['per_class'][1]['precision'], 0.0)

if __name__ == '__main__':
    unittest.main()
