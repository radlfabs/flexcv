from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from flexcv.metrics import MetricsDict


def test_metrics_dict_init():
    # Test initialization
    metrics = MetricsDict()
    assert isinstance(metrics, MetricsDict)
    assert metrics["r2"] == r2_score
    assert metrics["mse"] == mean_squared_error
    assert metrics["mae"] == mean_absolute_error


def test_metrics_dict_custom_metric():
    # Test adding a custom metric
    def custom_metric(y_true, y_pred):
        return sum(y_true) - sum(y_pred)

    metrics = MetricsDict()
    metrics["custom"] = custom_metric
    assert metrics["custom"] == custom_metric


def test_metrics_dict_metric_calculation():
    # Test calculation of a metric
    def custom_metric(y_true, y_pred):
        return sum(y_true) - sum(y_pred)

    metrics = MetricsDict()
    metrics["custom"] = custom_metric
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert metrics["custom"](y_true, y_pred) == 0
