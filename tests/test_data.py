import unittest

import numpy as np
import pandas as pd

from src.benchmark import validate_data


class DataValidationTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"Time": [0.0, 1.0], "Amount": [2.0, 3.0], "Class": [0, 1]})

    def test_valid_binary_labels(self):
        features, labels = validate_data(self.frame())
        self.assertNotIn("Class", features.columns)
        self.assertEqual(labels.tolist(), [0, 1])

    def test_fractional_labels_are_not_silently_truncated(self):
        data = self.frame()
        data["Class"] = [0.5, 1.5]
        with self.assertRaisesRegex(ValueError, "binary labels"):
            validate_data(data)

    def test_invalid_features(self):
        for value in (np.nan, np.inf, -np.inf, "invalid"):
            with self.subTest(value=value):
                data = self.frame()
                data["Amount"] = [2.0, value]
                with self.assertRaises(ValueError):
                    validate_data(data)

    def test_missing_schema_and_single_class(self):
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            validate_data(self.frame().drop(columns="Amount"))
        data = self.frame()
        data["Class"] = 0
        with self.assertRaisesRegex(ValueError, "binary labels"):
            validate_data(data)
