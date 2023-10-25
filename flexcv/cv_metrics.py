import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse_score(y_true, y_pred):
    """This function calculates the root mean squared error (RMSE) between the true and predicted values.
    It takes the squared root (numpy) of the sklearn MSE function."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse_wrapper(y_valid, y_pred, y_train_in, y_pred_train):
    """This function is only used to calculate the objective function value for the hyperparameter optimization.
    In order to allow for customized objective functions it takes the validation and training data and the corresponding predictions as arguments.
    This can be useful to avoid overfitting. The sklearn MSE function had to be wrapped accordingly
    """
    return mean_squared_error(y_valid, y_pred)


class MetricsDict(dict):
    """A dictionary that maps metric names to functions.
    It can be passed to the cross_validate function to specify which metrics to calculate in the outer loop.
    """

    def __init__(self):
        super().__init__()
        self["r2"] = r2_score
        self["mse"] = mean_squared_error
        self["mae"] = mean_absolute_error
        self["rmse"] = rmse_score


METRICS = MetricsDict()
