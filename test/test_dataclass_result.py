import pandas as pd
import pytest
from flexcv.fold_logging import SingleModelFoldResult

def test_single_model_fold_result_init():
    # Test initialization with valid arguments
    result = SingleModelFoldResult(
        k=1,
        model_name="TestModel",
        best_model=object(),
        best_params={"param": "value"},
        y_pred=pd.Series([1, 2, 3]),
        y_test=pd.Series([1, 2, 3]),
        X_test=pd.DataFrame({"column1": [1, 2, 3]}),
        y_train=pd.Series([1, 2, 3]),
        y_pred_train=pd.Series([1, 2, 3]),
        X_train=pd.DataFrame({"column1": [1, 2, 3]}),
        fit_result=object()
    )

    # Check that all attributes are set correctly
    assert result.k == 1
    assert result.model_name == "TestModel"
    assert isinstance(result.best_model, object)
    assert result.best_params == {"param": "value"}
    assert result.y_pred.equals(pd.Series([1, 2, 3]))
    assert result.y_test.equals(pd.Series([1, 2, 3]))
    assert result.X_test.equals(pd.DataFrame({"column1": [1, 2, 3]}))
    assert result.y_train.equals(pd.Series([1, 2, 3]))
    assert result.y_pred_train.equals(pd.Series([1, 2, 3]))
    assert result.X_train.equals(pd.DataFrame({"column1": [1, 2, 3]}))
    assert isinstance(result.fit_result, object)
