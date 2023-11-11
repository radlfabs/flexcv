import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from flexcv.fold_logging import log_single_model_single_fold

class TestLogSingleModelSingleFold(unittest.TestCase):
    def setUp(self):
        self.y_test = np.array([1, 2, 3])
        self.y_pred = np.array([1, 2, 3])
        self.y_train = np.array([1, 2, 3])
        self.y_pred_train = np.array([1, 2, 3])
        self.model_name = "TestModel"
        self.best_model = RandomForestRegressor()
        self.best_params = {"param": "value"}
        self.k = 1
        self.run = MagicMock()
        self.results_all_folds = {}
        self.study = MagicMock()

    @patch("flexcv.fold_logging.plt.figure")
    def test_log_single_model_single_fold(self, mock_figure):
        result = log_single_model_single_fold(
            self.y_test,
            self.y_pred,
            self.y_train,
            self.y_pred_train,
            self.model_name,
            self.best_model,
            self.best_params,
            self.k,
            self.run,
            self.results_all_folds,
            self.study
        )

        # Check that the results dictionary is updated correctly
        self.assertIn(self.model_name, result)
        self.assertIn("model", result[self.model_name])
        self.assertIn("parameters", result[self.model_name])
        self.assertIn("metrics", result[self.model_name])
        self.assertIn("folds_by_metrics", result[self.model_name])
        self.assertIn("y_pred", result[self.model_name])
        self.assertIn("y_test", result[self.model_name])
        self.assertIn("y_pred_train", result[self.model_name])
        self.assertIn("y_train", result[self.model_name])

        # Check that the metrics are calculated correctly
        self.assertEqual(result[self.model_name]["metrics"][0]["r2"], 1.0)
        self.assertEqual(result[self.model_name]["metrics"][0]["mse"], 0.0)
        self.assertEqual(result[self.model_name]["metrics"][0]["mae"], 0.0)

        self.assertIsInstance(result, dict)
        self.assertIsInstance(result[self.model_name]["model"][0], RandomForestRegressor)
        self.assertIsInstance(result[self.model_name]["parameters"][0], dict)
        self.assertIsInstance(result[self.model_name]["metrics"][0], dict)
        self.assertIsInstance(result[self.model_name]["folds_by_metrics"], dict)

if __name__ == "__main__":
    unittest.main()