import pandas as pd
import numpy as np
from flexcv.results_handling import add_summary_stats
from flexcv.results_handling import CrossValidationResults

def test_add_summary_stats():
    # Test add_summary_stats function with a DataFrame of numeric values
    df = pd.DataFrame({'column1': [1, 2, 3], 'column2': [4, 5, 6]})
    result = add_summary_stats(df)
    expected_result = pd.DataFrame({'column1': [1, 2, 3, 2, 2, 1], 'column2': [4, 5, 6, 5, 5, 1]}, index=[0, 1, 2, 'mean', 'median', 'std'])
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)

def test_cross_validation_results_init():
    # Test initialization
    results_dict = {
        "Model1": {
            "metrics": [
                {"metric1": 1, "metric2": 2},
                {"metric1": 3, "metric2": 4}
            ]
        },
        "Model2": {
            "metrics": [
                {"metric1": 5, "metric2": 6},
                {"metric1": 7, "metric2": 8}
            ]
        }
    }
    results = CrossValidationResults(results_dict)
    assert isinstance(results, CrossValidationResults)
    assert results == results_dict

# TODO add tests for results.summary generation

def test_get_best_model_by_metric_min():
    # Test get_best_model_by_metric method with direction="min"
    results_dict = {
        "Model1": {
            "metrics": {"mse": [1, 2, 3]},
            "model": ["model1_v1", "model1_v2", "model1_v3"]
        },
        "Model2": {
            "metrics": {"mse": [4, 5, 6]},
            "model": ["model2_v1", "model2_v2", "model2_v3"]
        }
    }
    results = CrossValidationResults(results_dict)
    best_model = results.get_best_model_by_metric(metric_name="mse", direction="min")
    assert best_model == "model1_v1"

def test_get_best_model_by_metric_max():
    # Test get_best_model_by_metric method with direction="max"
    results_dict = {
        "Model1": {
            "metrics": {"r2": [0.1, 0.2, 0.3]},
            "model": ["model1_v1", "model1_v2", "model1_v3"]
        },
        "Model2": {
            "metrics": {"r2": [0.4, 0.5, 0.6]},
            "model": ["model2_v1", "model2_v2", "model2_v3"]
        }
    }
    results = CrossValidationResults(results_dict)
    best_model = results.get_best_model_by_metric(metric_name="r2", direction="max")
    assert best_model == "model2_v3"

def test_get_best_model_by_metric_specific_model():
    # Test get_best_model_by_metric method with a specific model
    results_dict = {
        "Model1": {
            "metrics": {"mse": [1, 2, 3]},
            "model": ["model1_v1", "model1_v2", "model1_v3"]
        },
        "Model2": {
            "metrics": {"mse": [4, 5, 6]},
            "model": ["model2_v1", "model2_v2", "model2_v3"]
        }
    }
    results = CrossValidationResults(results_dict)
    best_model = results.get_best_model_by_metric(model_name="Model2", metric_name="mse", direction="min")
    assert best_model == "model2_v1"
    
def test_get_predictions():
    # Test get_predictions method
    results_dict = {
        "Model1": {
            "y_pred": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }
    }
    results = CrossValidationResults(results_dict)
    predictions = results.get_predictions(model_name="Model1", fold_id=1)
    assert predictions == [4, 5, 6]

def test_get_true_values():
    # Test get_true_values method
    results_dict = {
        "Model1": {
            "y_test": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }
    }
    results = CrossValidationResults(results_dict)
    true_values = results.get_true_values(model_name="Model1", fold_id=1)
    assert true_values == [4, 5, 6]

def test_get_training_predictions():
    # Test get_training_predictions method
    results_dict = {
        "Model1": {
            "y_pred_train": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }
    }
    results = CrossValidationResults(results_dict)
    training_predictions = results.get_training_predictions(model_name="Model1", fold_id=1)
    assert training_predictions == [4, 5, 6]

def test_get_training_true_values():
    # Test get_training_true_values method
    results_dict = {
        "Model1": {
            "y_train": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }
    }
    results = CrossValidationResults(results_dict)
    training_true_values = results.get_training_true_values(model_name="Model1", fold_id=1)
    assert training_true_values == [4, 5, 6]

def test_get_params():
    # Test get_params method
    results_dict = {
        "Model1": {
            "parameters": [{"param1": 1, "param2": 2}, {"param1": 3, "param2": 4}, {"param1": 5, "param2": 6}]
        }
    }
    results = CrossValidationResults(results_dict)
    params = results.get_params(model_name="Model1", fold_id=1)
    assert params == {"param1": 3, "param2": 4}