from flexcv.model_postprocessing import LinearModelPostProcessor, LMERModelPostProcessor
from flexcv.run import Run
from flexcv.fold_results_handling import SingleModelFoldResult
from flexcv.models import LinearModel, LinearMixedEffectsModel
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from unittest.mock import MagicMock, patch

def test_linear_model_postprocessor_init():
    # Test initialization
    linear_model_postprocessor = LinearModelPostProcessor()
    assert isinstance(linear_model_postprocessor, LinearModelPostProcessor)

@patch("flexcv.model_postprocessing.File.from_content")
def test_lmer_model_postprocessor_call(mock_from_content):
    # Test __call__ method
    lmer_model_postprocessor = LMERModelPostProcessor()
    results_all_folds = {"LMER": {"parameters": {}}}
    k = 1
    model_name = "LMER"
    best_model = LinearMixedEffectsModel()
    best_params = {"fit_intercept": True}
    y_pred = pd.Series(np.random.rand(10))
    y_test = pd.Series(np.random.rand(10))
    X_test = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series(np.random.rand(10))
    y_pred_train = pd.Series(np.random.rand(10))
    X_train = pd.DataFrame(np.random.rand(10, 5))
    fit_result = MagicMock()
    fit_result.get_summary.return_value = "summary"
    fold_result = SingleModelFoldResult(
        k,
        model_name,
        best_model,
        best_params,
        y_pred,
        y_test,
        X_test,
        y_train,
        y_pred_train,
        X_train,
        fit_result,
    )
    features = ["feature1", "feature2", "feature3"]
    run = Run()
    results = lmer_model_postprocessor(results_all_folds, fold_result, features, run)
    assert isinstance(results, dict)
    assert model_name in results
    assert "parameters" in results[model_name]
    assert results[model_name]["parameters"][k] == "summary"
    mock_from_content.assert_called_once_with("summary", extension="html")