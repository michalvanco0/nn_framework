import unittest

from evaluation.performance import calculate_classification_metrics


class TestPerformanceMetrics(unittest.TestCase):
    def test_calculate_classification_metrics_outputs_expected_fields(self) -> None:
        y_true = [0, 1, 2, 1]
        y_pred = [0, 1, 0, 1]

        result = calculate_classification_metrics(y_true, y_pred)

        self.assertEqual(
            set(result.keys()),
            {"accuracy", "precision", "recall", "f1", "confusion_matrix"},
        )
        self.assertTrue(0.0 <= result["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= result["precision"] <= 1.0)
        self.assertTrue(0.0 <= result["recall"] <= 1.0)
        self.assertTrue(0.0 <= result["f1"] <= 1.0)
        self.assertEqual(len(result["confusion_matrix"]), 3)
