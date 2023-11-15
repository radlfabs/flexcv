"""
This module provides a template dictionary that maps mutiple machine learning model names to the respective model configuration dicts.
"""
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from . import model_postprocessing as mp
from .merf import MERF
from .model_mapping import ModelConfigDict, ModelMappingDict
from .models import LinearMixedEffectsModel, LinearModel

MODEL_MAPPING = ModelMappingDict(
    {
        "LinearModel": ModelConfigDict(
            {
                "requires_inner_cv": False,
                "n_trials": 100,
                "n_jobs_model": 1,
                "n_jobs_cv": 1,
                "model": LinearModel,
                "params": {},
                "post_processor": mp.LinearModelPostProcessor,
            }
        ),
        "RandomForest": ModelConfigDict(
            {
                "requires_inner_cv": True,
                "n_trials": 400,
                "n_jobs_model": 1,
                "n_jobs_cv": -1,
                "model": RandomForestRegressor,
                "params": {
                    "max_depth": optuna.distributions.IntDistribution(5, 100),
                    "min_samples_split": optuna.distributions.IntDistribution(
                        2, 1000, log=True
                    ),
                    "min_samples_leaf": optuna.distributions.IntDistribution(
                        2, 5000, log=True
                    ),
                    "max_samples": optuna.distributions.FloatDistribution(0.0021, 0.9),
                    "max_features": optuna.distributions.IntDistribution(1, 10),
                    "max_leaf_nodes": optuna.distributions.IntDistribution(10, 40000),
                    "min_impurity_decrease": optuna.distributions.FloatDistribution(
                        1e-8, 0.02, log=True
                    ),  # >>>> can be (1e-8, .01, log=True)
                    "min_weight_fraction_leaf": optuna.distributions.FloatDistribution(
                        0, 0.5
                    ),  # must be a float in the range [0.0, 0.5]
                    "ccp_alpha": optuna.distributions.FloatDistribution(1e-8, 0.01),
                    "n_estimators": optuna.distributions.IntDistribution(2, 7000),
                },
                "post_processor": mp.RandomForestModelPostProcessor,
                "add_merf": True,
            }
        ),
        "XGBoost": ModelConfigDict(
            {
                "requires_inner_cv": True,
                "n_trials": 300,
                "n_jobs_model": 1,
                "n_jobs_cv": -1,
                "model": XGBRegressor,
                "params": {
                    "max_depth": optuna.distributions.IntDistribution(2, 700),
                    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.8),
                    "n_estimators": optuna.distributions.IntDistribution(5, 5000),
                    "min_child_weight": optuna.distributions.IntDistribution(2, 100),
                    # "max_delta_step": optuna.distributions.FloatDistribution(0.1, 10.0),
                    # "gamma": optuna.distributions.FloatDistribution(0.0, 50),
                    "subsample": optuna.distributions.FloatDistribution(0.005, 0.97),
                    "colsample_bytree": optuna.distributions.FloatDistribution(
                        0.1, 1, step=0.1
                    ),
                    "colsample_bylevel": optuna.distributions.FloatDistribution(
                        0.1, 1, step=0.1
                    ),
                    "colsample_bynode": optuna.distributions.FloatDistribution(
                        0.1, 1, step=0.1
                    ),
                    "colsample_bytree": optuna.distributions.FloatDistribution(
                        0.1, 1, step=0.1
                    ),
                    "colsample_bylevel": optuna.distributions.FloatDistribution(
                        0.1, 1, step=0.1
                    ),
                    "colsample_bynode": optuna.distributions.FloatDistribution(
                        0.1, 1, step=0.1
                    ),
                    "reg_alpha": optuna.distributions.FloatDistribution(0.1, 500),
                    "reg_lambda": optuna.distributions.FloatDistribution(0.001, 800),
                },
                "post_processor": mp.XGBoostModelPostProcessor,
                "add_merf": True,
            }
        ),
        "SVR": ModelConfigDict(
            {
                "requires_inner_cv": True,
                "n_trials": 450,
                "allows_n_jobs": False,
                "n_jobs_cv": -1,
                "model": SVR,
                "params": {
                    # Most Important: Kernel + C
                    # "kernel": default "rbf" yielded best results
                    # "degree": # for poly only
                    "C": optuna.distributions.FloatDistribution(0.001, 50, log=True),
                    "epsilon": optuna.distributions.FloatDistribution(0.1, 1.3),
                    "gamma": optuna.distributions.FloatDistribution(
                        1e-5, 0.1, log=True
                    ),  # better than default "scale"
                    # "tol": optuna.distributions.FloatDistribution(1e-4, 10),
                    # "shrinking": default "True" yielded best restults
                },
                "post_processor": mp.SVRModelPostProcessor,
                "add_merf": True,
            }
        ),
    }
)
