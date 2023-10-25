import logging
import gc
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

np.random.seed(42)
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)


def log_funcname(func):
    @wraps(func)
    def wrapper_function(*args, **kwargs):
        logger.info(f"Calling {func.__name__} model function.")
        results = func(*args, **kwargs)
        return results

    return wrapper_function


class BaseLinearModel(BaseEstimator, RegressorMixin):
    def __init__(self, re_formula=None, verbose=0, *args, **kwargs):
        self.re_formula = re_formula
        self.verbose = verbose
        self.best_params = {}
        self.params = {}

    def get_params(self, deep=True):
        return self.params

    def get_summary(self):
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

    def fit(self, X, y, **kwargs):
        """
        Fit the LMER_LLS model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        **kwargs : dict
            Additional parameters to pass to the underlying model's `fit` method.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        This method fits a OLS class on the X data.
        """
        # X, y = check_X_y(X, y)
        assert (
            X.shape[0] == y.shape[0]
        ), "Number of X samples must match number of y samples."
        assert type(X) == pd.DataFrame, "X must be a pandas DataFrame."
        assert type(y) == pd.Series, "y must be a pandas Series."

        self.X_ = X
        self.y_ = y

        data = pd.concat([y, X], axis=1, sort=False)
        data.columns = [y.name] + list(X.columns)
        md = smf.ols(kwargs["formula"], data)
        self.md_ = md.fit()
        self.best_params = self.get_summary()
        return self

    def predict(self, X, **kwargs):
        """Returns
        -------
        An array of fitted values.  Note that these predicted values
        only reflect the fixed effects mean structure of the model.
        """
        check_is_fitted(self, ["X_", "y_", "md_"])
        return self.md_.predict(exog=X)


class LinearMixedEffectsModel(BaseLinearModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, re_formula, **kwargs):
        """
        Fit the LMER_LLS model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        clusters : array-like of shape (n_samples,)
        **kwargs : dict
            Additional parameters to pass to the underlying model's `fit` method.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        This method fits a MixedLM to the data.
        """
        # X, y = check_X_y(X, y)
        assert (
            X.shape[0] == y.shape[0]
        ), "Number of X samples must match number of y samples."
        assert type(X) == pd.DataFrame, "X must be a pandas DataFrame."
        assert type(y) == pd.Series, "y must be a pandas Series."
        assert "clusters" in kwargs, "clusters must be present in kwargs."
        assert (
            len(kwargs["clusters"]) == X.shape[0]
        ), "Number of clusters must match number of samples."

        self.X_ = X
        self.y_ = y
        self.cluster_counts = kwargs["clusters"].value_counts()
        self.re_formula = re_formula
        data = pd.concat([y, X, kwargs["clusters"]], axis=1, sort=False)
        data.columns = [y.name] + list(X.columns) + [kwargs["clusters"].name]
        # if re_formula is None we pass a empty dict, else we pass the re_formula
        re_formula_dict = {"re_formula": re_formula} if self.re_formula else {}
        md = smf.mixedlm(
            formula=kwargs["formula"],
            data=data,
            groups=kwargs["clusters"].name,
            **re_formula_dict,
        )

        self.md = md
        self.md_ = md.fit()

        self.best_params = self.get_summary()
        return self

    def predict(self, X, **kwargs):
        """Returns
        -------
        An array of fitted values.  Note that these predicted values
        only reflect the fixed effects mean structure of the model.
        """
        check_is_fitted(self, ["X_", "y_", "md_"])
        clusters = kwargs["clusters"]
        predict_known_groups_lmm = kwargs["predict_known_groups_lmm"]
        Z = kwargs["Z"]
        assert (
            len(clusters) == X.shape[0]
        ), "Number of clusters must match number of samples."

        if predict_known_groups_lmm == True:
            yp = self.md_.predict(exog=X)

            for cluster_id in self.cluster_counts.index:
                indices_i = clusters == cluster_id

                # If cluster doesn't exist move on.
                if len(indices_i) == 0:
                    continue

                # # If cluster does exist, apply the correction.
                b_i = self.md_.random_effects[cluster_id]

                # Z_i = self.md.exog_re_li[indices_i]
                Z_i = Z[indices_i]
                yp[indices_i] += Z_i.dot(b_i)

            return yp
        else:
            return self.md_.predict(exog=X)


class EarthRegressor(BaseEstimator, RegressorMixin):
    """Wrapper Class for Earth Regressor in R.
    More Details see https://cran.r-project.org/web/packages/earth/earth.pdf.

    Hyperparameters:
        degree: int, default=1
            Degree of the splines. 1 for linear, 2 for quadratic, etc.
        nprune: int, default=None
            Number of pruning steps. If None, the number of pruning steps is determined by the algorithm.
        nk: int, default=None
            Number of knots. If None, the number of knots is determined by the algorithm.
            The default is semi-automatically calculated from the number of predictors but may need adjusting.
        thresh: float, default=0.001
            Forward stepping threshold.
        minspan: int, default=0
            Minimum number of observations between knots.
        endspan: int, default=0
            Minimum number of observations before the first and after the final knot.
        newvar_penalty: float, default=0.0
        fast_k: int, default=20
            Maximum number of parent terms considered at each step of the forward pass.
        fast_beta: float, default=1.0
            Fast MARS ageing coefficient, as described in the Fast MARS paper section 3.1.
            Default is 1. A value of 0 sometimes gives better results.
        pmethod: str, default="backward"
            Pruning method. One of: backward none exhaustive forward seqrep cv.
            Default is "backward". Specify pmethod="cv" to use cross-validation to select the number of terms.
            This selects the number of terms that gives the maximum mean out-of-fold RSq on the fold models.
            Requires the nfold argument. Use "none" to retain all the terms created by the forward pass.
            If y has multiple columns, then only "backward" or "none" is allowed.
            Pruning can take a while if "exhaustive" is chosen and the model is big (more than about 30 terms).
            The current version of the leaps package used during pruning does not allow user interrupts
            (i.e., you have to kill your R session to interrupt; in Windows use the Task Manager or from the command line use taskkill).
    """

    def __init__(
        self,
        degree: int = 1,
        nprune: int = None,
        nk: int = None,
        thresh: float = 0.001,
        minspan: int = 0,
        endspan: int = 0,
        newvar_penalty: float = 0.0,
        fast_k: int = 20,
        fast_beta: float = 1.0,
        pmethod: str = "backward",
        random_state: int = None,
    ):
        self.degree = degree
        self.nprune = nprune
        self.nk = nk
        self.thresh = thresh
        self.minspan = minspan
        self.endspan = endspan
        self.newvar_penalty = newvar_penalty
        self.fast_k = fast_k
        self.fast_beta = fast_beta
        self.pmethod = pmethod
        self.random_state = random_state

    def fit(self, X, y):
        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("Complex data not supported")
        # ro.r('sink(nullfile())')
        if self.random_state is not None:
            ro.r(f"set.seed({self.random_state})")

        ro.r(
            """
            library(earth)
        """
        )
        numpy2ri.activate()
        pandas2ri.activate()

        # assert type(X) == pd.DataFrame, "X must be a pandas DataFrame."
        # assert type(y) == pd.Series, "y must be a pandas Series."
        assert (
            X.shape[0] == y.shape[0]
        ), "Number of X samples must match number of y samples."

        # Convert X, y according to its type
        if isinstance(X, pd.DataFrame):
            # Convert pandas dataframe to R dataframe
            r_X = pandas2ri.py2rpy(X)
        elif isinstance(X, np.ndarray):
            r_X = numpy2ri.numpy2rpy(X)
            # Convert numpy array to R matrix
        else:
            r_X = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])

        # Convert pandas Series to R vector
        r_y = ro.FloatVector(y)

        # Fit MARS regression model using earth function from the earth package
        # make nprune None in R as default
        nprune = self.nprune if self.nprune is not None else ro.r("as.null")()
        # The following has a special defaults which we dont want to overwrite with None
        nk = {"nk": self.nk} if self.nk is not None else {}
        # We have to pass newvar.penalty as a named argument because Python does not allow "." in variable names
        newvar_penalty = {"newvar.penalty": self.newvar_penalty}
        fast_k = {"fast.k": self.fast_k}
        fast_beta = {"fast.beta": self.fast_beta}

        self.model_ = ro.r.earth(
            r_X,
            r_y,
            degree=self.degree,
            nprune=nprune,
            thresh=self.thresh,
            minspan=self.minspan,
            endspan=self.endspan,
            pmethod=self.pmethod,
            **newvar_penalty,
            **fast_k,
            **fast_beta,
            **nk,
        )

        self.is_fitted_ = True
        self.var_imp_ = self.calc_variable_importance()

        del r_X
        del r_y
        numpy2ri.deactivate()
        pandas2ri.deactivate()

        gc.collect()
        ro.r("gc()")
        gc.collect()

        return self

    def predict(self, X):
        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")
        # ro.r('sink(nullfile())')
        ro.r(
            """
            library(earth)
        """
        )
        numpy2ri.activate()
        pandas2ri.activate()
        # input checks
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            # Convert pandas dataframe to R dataframe
            r_X = pandas2ri.py2rpy(X)
        elif isinstance(X, np.ndarray):
            r_X = numpy2ri.numpy2rpy(X)
            # Convert numpy array to R matrix
        else:
            r_X = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        # assign model in R in order to predict
        # ro.r.assign("model", self.model)
        y_pred = np.asarray(ro.r["predict"](self.model_, r_X))
        # make sure that the output is a 1d array
        y_pred = y_pred.ravel()
        del r_X

        numpy2ri.deactivate()
        pandas2ri.deactivate()
        gc.collect()
        ro.r("gc()")
        gc.collect()

        return y_pred

    def __sklearn_is_fitted__(self):
        return self.is_fitted_

    def get_params(self, deep=False):
        return {
            "degree": self.degree,
            "nprune": self.nprune,
            "nk": self.nk,
            "thresh": self.thresh,
            "minspan": self.minspan,
            "endspan": self.endspan,
            "newvar_penalty": self.newvar_penalty,
            "fast_k": self.fast_k,
            "fast_beta": self.fast_beta,
            "pmethod": self.pmethod,
            "random_state": self.random_state,
        }

    def get_rmodel(self):
        return self.model_

    def make_r_plots(self):
        Path("tmp_imgs").mkdir(parents=True, exist_ok=True)
        for i in range(1, 5):
            ro.r["png"](f"tmp_imgs/mars_plot_{i}.png", width=1024, height=1024)
            ro.r["plot"](self.model_, which=i)
            ro.r["dev.off"]()

    def calc_variable_importance(self):
        ro.globalenv["ev"] = ro.r["evimp"](self.model_, trim=False)
        imp = ro.r("as.data.frame(unclass(ev[,c(3,4,6)]))")
        imp_df: pd.DataFrame = ro.conversion.rpy2py(imp)
        imp_df.columns = ["nsubsets", "gcv", "rss"]

        del imp
        gc.collect()
        ro.r("rm(ev)")
        ro.r("gc()")
        gc.collect()
        return imp_df

    def get_variable_importance(self, features):
        self.var_imp_.index = features
        return self.var_imp_


if __name__ == "__main__":
    earth = EarthRegressor()
    check_estimator(earth)
