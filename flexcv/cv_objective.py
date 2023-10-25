from typing import Callable
import inspect
import multiprocessing
import numpy as np
from sklearn.metrics import mean_squared_error


class ObjectiveScorer(
    Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
):
    """
    Callable class that wraps a scorer function to be used as an objective function.
    The scorer function must take the following arguments:
    y_valid: ndarray
        The validation target values.
    y_pred: ndarray
        The predicted target values.
    y_train_in: ndarray
        The training target values.
    y_pred_train: ndarray
        The predicted training target values.
    and return a float value.
    """

    def __init__(
        self, scorer: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
    ):
        self.scorer = scorer
        self.check_signature()

    def __call__(
        self,
        y_valid: np.ndarray,
        y_pred: np.ndarray,
        y_train_in: np.ndarray,
        y_pred_train: np.ndarray,
    ) -> float:
        return self.scorer(y_valid, y_pred, y_train_in, y_pred_train)

    def check_signature(self):
        expected_args = ["y_valid", "y_pred", "y_train_in", "y_pred_train"]
        signature = inspect.signature(self.scorer)
        for arg_name, param in signature.parameters.items():
            if arg_name not in expected_args:
                raise ValueError(
                    f"Invalid argument name '{arg_name}' in scorer function signature."
                )
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise ValueError(
                    f"Invalid parameter kind '{param.kind}' in scorer function signature."
                )
        if len(signature.parameters) != len(expected_args):
            raise ValueError(
                f"Invalid number of arguments in scorer function signature. Expected {len(expected_args)}, got {len(signature.parameters)}."
            )


def custom_scorer(y_valid, y_pred, y_train_in, y_pred_train) -> float:
    """Objective scorer for the hyperparameter optimization.
    The function calculates the mean squared error (MSE) for both the validation and training data, 
    and then calculates a weighted sum of the MSEs and their differences.
    The weights and thresholds used in the calculation are defined in the function.
    The function returns a float value that represents the objective function value.
    This function is used in the hyperparameter optimization process to evaluate the performance of different models with different hyperparameters.
    """

    mse_valid = mean_squared_error(y_valid, y_pred)
    mse_train = mean_squared_error(y_train_in, y_pred_train)

    mse_delta = mse_train - mse_valid
    target_delta = 0.05

    return (
        1 * mse_valid
        + 0.5 * abs(mse_delta)
        + 2 * max(0, (mse_delta - target_delta))
        + 1 * max(0, -mse_delta)
    )


def objective(
    X_train_in,
    y_train_in,
    X_valid,
    y_valid,
    pipe,
    params,
    objective_scorer: ObjectiveScorer,
):
    """Objective function for the hyperparameter optimization.
    Sets the parameters of the pipeline and fits it to the training data.
    Predicts the validation data and calculates the MSE for both the validation and training data.
    Then applies the objective scorer to the validation MSE and the training MSE which returns the objective function value.
    Returns the negative validation and training MSEs as well as the negative objective function value, since optuna maximizes the objective function.
    """

    pipe.set_params(**params)

    pipe.fit(X_train_in, y_train_in)

    y_pred = pipe.predict(X_valid)
    y_pred_train = pipe.predict(X_train_in)

    score_valid = mean_squared_error(y_valid, y_pred)
    score_train = mean_squared_error(y_train_in, y_pred_train)
    score_of = objective_scorer(y_valid, y_pred, y_train_in, y_pred_train)

    return -score_valid, -score_train, -score_of


def parallel_objective(
    train_idx, valid_idx, X, y, pipe, params_, objective_scorer: ObjectiveScorer
):
    """Parallel objective function for the hyperparameter optimization.
    Gets the training and validation indices and the data and calls the objective function.
    """
    X_train_in = X.iloc[train_idx]
    y_train_in = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    score_valid, score_train, score_OF = objective(
        X_train_in, y_train_in, X_valid, y_valid, pipe, params_, objective_scorer
    )

    return score_valid, score_train, score_OF


def objective_cv(
    trial, cross_val_split, pipe, params, X, y, run, n_jobs, objective_scorer
):
    """Objective function for the hyperparameter optimization with cross validation.
    n_jobs is the number of processes to use for the parallelization.
    If n_jobs is -1, the number of processes is set to the number of available CPUs.
    If n_jobs is 1, the objective function is called sequentially."""

    params_ = {
        name: trial._suggest(name, distribution)
        for name, distribution in params.items()
    }

    scores_valid = []
    scores_train = []
    scores_OF = []

    if n_jobs == -1:
        # Define the number of processes to use and create a pool
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        # Map the parallel function to the cross validation split
        results = pool.starmap(
            parallel_objective,
            [
                (train_idx, valid_idx, X, y, pipe, params_, objective_scorer)
                for train_idx, valid_idx in cross_val_split(X=X, y=y)
            ],
        )
        pool.close()

        for result in results:
            scores_valid.append(result[0])
            scores_train.append(result[1])
            scores_OF.append(result[2])
    else:
        for train_idx, valid_idx in cross_val_split(X=X, y=y):
            X_train_in = X.iloc[train_idx]
            y_train_in = y.iloc[train_idx]

            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]

            score_valid, score_train, score_OF = objective(
                X_train_in, y_train_in, X_valid, y_valid, pipe, params_
            )

            scores_valid.append(score_valid)
            scores_train.append(score_train)
            scores_OF.append(score_OF)

    trial.set_user_attr("mean_test_score", np.mean(scores_valid))
    trial.set_user_attr("mean_train_score", np.mean(scores_train))
    trial.set_user_attr("mean_OF_score", np.mean(scores_OF))

    return np.mean(scores_OF)
