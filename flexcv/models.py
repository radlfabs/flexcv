"""This module implements wrapper classes for the Linear Model and the Linear Mixed Effects Model from statsmodels.
"""

import gc
import logging
import warnings
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)


class BaseLinearModel(BaseEstimator, RegressorMixin):
    """Base class for the Linear Model and the Linear Mixed Effects Model."""

    def __init__(self, re_formula=None, verbose=0, *args, **kwargs):
        self.re_formula = re_formula
        self.verbose = verbose
        self.best_params = {}
        self.params = {}

    def get_params(self, deep=True):
        """Return the parameters of the model.

        Args:
          deep: This argument is not used. (Default value = True)

        Returns:
            (dict): Parameter names mapped to their values.

        """
        return self.params

    def get_summary(self):
        """Creates a html summary table of the model.

        Returns:
            (str): HTML table of the model summary."""
        lmer_summary = self.md_.summary()  # type: ignore
        try:
            html_tables = ""
            for table in lmer_summary.tables:
                html_tables += table.as_html()
        except AttributeError:
            html_tables = lmer_summary.as_html()
        return html_tables


class LinearModel(BaseLinearModel):
    """Wrapper class for the Linear Model from statsmodels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, formula, **kwargs):
        """Fit the LM to the given training data.

        Args:
          X (array-like of shape (n_samples, n_features)): The training input samples.
          y (array-like of shape (n_samples,)): The target values.
          **kwargs(dict): Additional parameters to pass to the underlying model's `fit` method.

        Returns:
          (object): Returns the model after fit.

        Notes:
            This method fits a OLS class on the X data.
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "Number of X samples must match number of y samples."
        assert type(X) == pd.DataFrame, "X must be a pandas DataFrame."
        assert type(y) == pd.Series, "y must be a pandas Series."

        self.X_ = X
        self.y_ = y

        data = pd.concat([y, X], axis=1, sort=False)
        data.columns = [y.name] + list(X.columns)
        md = smf.ols(formula, data)
        self.md_ = md.fit()
        self.best_params = self.get_summary()
        return self

    def predict(self, X, **kwargs):
        """Make predictions using the fitted model.

        Args:
          X (array-like): Features
          **kwargs: Used to prevent raising an error when passing the `clusters` argument.

        Returns:
          (array-like): An array of fitted values.


        """
        check_is_fitted(self, ["X_", "y_", "md_"])
        return self.md_.predict(exog=X)


class LinearMixedEffectsModel(BaseLinearModel):
    """Wrapper class for the Linear Mixed Effects Model from statsmodels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, clusters, formula, re_formula, **kwargs):
        """Fit the LMER model to the given training data.

        Args:
          X (pd.DataFrame): The training input samples.
          y (pd.Series): The target values.
          clusters (pd.Series): The clustering data.
          re_formula (str): The random effects formula for the random slopes and intercepts.
          **kwargs (dict): Additional parameters to pass to the underlying model's `fit` method.


        Returns:
          (object): Returns self.

        Notes:
            This method fits a LMER class on the X data.
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "Number of X samples must match number of y samples."
        assert type(X) == pd.DataFrame, "X must be a pandas DataFrame."
        assert type(y) == pd.Series, "y must be a pandas Series."
        assert type(clusters) == pd.Series, "clusters must be a pandas Series."
        assert (
            len(y) == X.shape[0]
        ), "Number of target must match number of feature samples."
        assert (
            len(clusters) == X.shape[0]
        ), "Number of clusters must match number of feature samples."

        assert (
            re_formula is not None
        ), "re_formula must be specified for the LMER model."

        assert (
            len(clusters.unique()) > 1
        ), "Only one cluster found. There might be a problem with the cluster column."

        self.X_ = X
        self.y_ = y
        self.cluster_counts = clusters.value_counts()
        self.re_formula = re_formula
        data = pd.concat([y, X, clusters], axis=1, sort=False)
        data.columns = [y.name] + list(X.columns) + [clusters.name]
        # if re_formula is None we pass a empty dict, else we pass the re_formula
        re_formula_dict = {"re_formula": re_formula} if self.re_formula else {}
        md = smf.mixedlm(
            formula=formula,
            data=data,
            groups=clusters.name,
            **re_formula_dict,
        )

        self.md = md
        self.md_ = md.fit()

        self.best_params = self.get_summary()
        return self

    def predict(self, X: pd.DataFrame, clusters: pd.Series, **kwargs):
        """
        Make predictions using the fitted model.

        Args:
          X (pd.DataFrame): Features
          clusters (pd.Series): The clustering data.
          **kwargs: Any other keyword arguments to pass to the underlying model's `predict` method. This is necessary to prevent raising an error when passing the `clusters` argument.

        Returns:
          (array-like): An array of fitted values.

        """
        check_is_fitted(self, ["X_", "y_", "md_"])
        predict_known_groups_lmm = kwargs["predict_known_groups_lmm"]
        Z = kwargs["Z"]
        assert type(X) == pd.DataFrame, "X must be a pandas DataFrame."
        assert type(clusters) == pd.Series, "clusters must be a pandas Series."
        assert (
            len(clusters) == X.shape[0]
        ), "Number of clusters must match number of samples."
        if len(clusters.unique()) == 1:
            logger.warning(
                "Only one cluster found. There might be a problem with the cluster column."
            )

        if predict_known_groups_lmm == True:
            yp = self.md_.predict(exog=X)

            for cluster_id in self.cluster_counts.index:
                indices_i = clusters == cluster_id

                # If cluster doesn't exist move on.
                if len(indices_i) == 0:
                    continue

                # # If cluster does exist, apply the correction.
                b_i = self.md_.random_effects[cluster_id]

                Z_i = Z[indices_i]
                yp[indices_i] += Z_i.dot(b_i)

            return yp
        else:
            return self.md_.predict(exog=X)


if __name__ == "__main__":
    pass
