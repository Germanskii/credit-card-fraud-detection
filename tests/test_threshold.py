import unittest
import numpy as np
from sklearn.metrics import f1_score
from src.benchmark import choose_threshold


class ThresholdTest(unittest.TestCase):
    def test_validation_threshold_maximizes_f1(self):
        y = np.array([0, 0, 1, 0, 1, 1])
        prob = np.array([0.05, 0.2, 0.3, 0.4, 0.6, 0.9])
        threshold = choose_threshold(y, prob)
        expected = max(f1_score(y, prob >= t) for t in np.unique(prob))
        self.assertAlmostEqual(f1_score(y, prob >= threshold), expected)

    def test_tied_scores(self):
        self.assertEqual(choose_threshold([0, 1, 0, 1], [0.5] * 4), 0.5)
