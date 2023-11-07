"""
This module implements the MetricsDict class and it's default metrics.
It is used to specify which metrics to calculate in the outer loop of the cross-validation.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mse_wrapper(y_valid, y_pred, y_train_in, y_pred_train):
    """This function is only used to calculate the objective function value for the hyperparameter optimization.
    In order to allow for customized objective functions it takes the validation and training data and the corresponding predictions as arguments.
    This can be useful to avoid overfitting. The sklearn MSE function had to be wrapped accordingly

    Args:
      y_valid (array-like): Target in the validation set.
      y_pred (array-like): Predictions for the validation set.
      y_train_in (array-like): Target in the training set.
      y_pred_train (array-like): Predictions for the training set.

    Returns:
        (float): Mean squared error.

    """
    return mean_squared_error(y_valid, y_pred)


class MetricsDict(dict):
    """A dictionary that maps metric names to functions.
    It can be passed to the cross_validate function to specify which metrics to calculate in the outer loop.
    
    Default Metrics:
        By default, the following metrics are initialized:
        
        - R²: The coefficient of determination
        
        - MSE: Mean squared error
        
        - MAE: Mean absolute error

    Parameters:
        (dict): A dictionary that maps metric names to functions.
        
    Note: 
        We decided againt using the RMSE as a default metric, because we would run into trouble wherever we would average over it.
        RMSE should always be averaged as `sqrt(mean(MSE_values))` and not as `mean(sqrt(MSE_values))`.
        Also, the standard deviation would be calculated incorrectly if RMSE is included at this point.
    """

    def __init__(self):
        super().__init__()
        self["r2"] = r2_score
        self["mse"] = mean_squared_error
        self["mae"] = mean_absolute_error


METRICS = MetricsDict()


if __name__ == "__main__":
    pass
