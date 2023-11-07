import logging
import warnings
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from neptune.integrations.python_logger import NeptuneHandler
from neptune.metadata_containers.run import Run as NeptuneRun
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_random_state
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

from .cv_logging import (
    CustomNeptuneCallback,
    SingleModelFoldResult,
    log_diagnostics,
    log_single_model_single_fold,
)
from .metrics import MetricsDict, mse_wrapper
from .model_selection import ObjectiveScorer, objective_cv
from .split import CrossValMethod, make_cross_val_split
from .utilities import get_fixed_effects_formula, get_re_formula
from .model_mapping import ModelMappingDict
from .merf import MERF
from .models import LinearMixedEffectsModel

warnings.filterwarnings("ignore", module=r"matplotlib\..*")
warnings.filterwarnings("ignore", module=r"xgboost\..*")
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)


def cross_validate(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    dataset_name: str,
    run: NeptuneRun,
    groups: pd.Series,
    slopes: pd.DataFrame | pd.Series,
    split_out: CrossValMethod,
    split_in: CrossValMethod,
    break_cross_val: bool,
    scale_in: bool,
    scale_out: bool,
    n_splits_out: int,
    n_splits_in: int,
    random_seed: int,
    model_effects: str,
    n_trials: int,
    mapping: ModelMappingDict,
    metrics: MetricsDict,
    objective_scorer: ObjectiveScorer,
    em_max_iterations: int,
    em_stopping_threshold: float,
    em_stopping_window: int,
    predict_known_groups_lmm: bool,
    diagnostics: bool,
) -> Dict[str, Dict[str, list]]:
    """This function performs a cross-validation for a given regression formula, using one or a number of specified machine learning models and a configurable cross-validation method.

    Args:
      *: Use only keyword arguments for this function.
      X: pd.DataFrame: Features.
      y: pd.Series: Target.
      target_name: str: Custom target name.
      dataset_name: str: Custom dataset name.
      run: NeptuneRun: A Run object to log to.
      groups: pd.Series: The grouping or clustering variable.
      slopes: pd.DataFrame | pd.Series: Random slopes variable(s)
      split_out: CrossValMethod: Outer split strategy.
      split_in: CrossValMethod: Inner split strategy.
      break_cross_val: bool: If True, only the first outer fold is evaluated.
      scale_in: bool: If True, the features are scaled in the inner cross-validation to zero mean and unit variance. This works independently of the outer scaling.
      scale_out: bool: If True, the features are scaled in the outer cross-validation to zero mean and unit variance.
      n_splits_out: int: Number of outer cross-validation folds.
      n_splits_in: int: Number of inner cross-validation folds.
      random_seed: int: Seed for all random number generators.
      model_effects: str: If "fixed", only fixed effects are used. If "mixed", both fixed and random effects are used.
      n_trials: int: Number of trials for the inner cross-validation, i.e. the number of hyperparameter combinations to sample from the distributions.
      mapping: ModelMappingDict: The mapping providing model instances, hyperparameter distributions, and postprocessing functions.
      metrics: MetricsDict: A dict of metrics to be used as the evaluation metric for the outer cross-validation.
      objective_scorer: ObjectiveScorer: A custom objective scorer object to provide the evaluation metric for the inner cross-validation.
      em_max_iterations: int: For use with MERF. Maximum number of iterations for the EM algorithm.
      em_stopping_threshold: float: For use with MERF. Threshold for the early stopping criterion of the EM algorithm.
      em_stopping_window: int: For use with MERF. Window size for the early stopping criterion of the EM algorithm.
      predict_known_groups_lmm: bool: For use with Mixed Linear Models. If True, the model will predict the known groups in the test set.
      diagnostics: bool: If True, diagnostics plots are logged to Neptune.

    Returns:
        (Dict[str, Dict[str, list]]) : A dictionary containing the results of the cross-validation, organized by machine learning models.

    The function returns a nested dictionary with the following structure:
    Usage:
        ```python
        results_all_folds = {
        model_name: {
        "model": [],
        "parameters": [],
        "results": [],
        "r2": [],
        "y_pred": [],
        "y_test": [],
        "shap_values": [],
        "median_index": [],
        }
        ```

    """
    if objective_scorer is None:
        objective_scorer = ObjectiveScorer(mse_wrapper)
    else:
        objective_scorer = ObjectiveScorer(objective_scorer)

    try:
        npt_handler = NeptuneHandler(run=run)
        logger.addHandler(npt_handler)
    except TypeError:
        logger.warning(
            "No Neptune run object passed. Logging to Neptune will be disabled."
        )

    if isinstance(em_max_iterations, int):
        max_iterations = em_max_iterations

    if isinstance(em_stopping_window, int):
        em_window = em_stopping_window

    if isinstance(em_stopping_threshold, float):
        em_stopping_threshold = em_stopping_threshold

    print()
    re_formula = get_re_formula(slopes)
    formula = get_fixed_effects_formula(target_name, X)
    cross_val_split_out = make_cross_val_split(
        method=split_out, groups=groups, n_splits=n_splits_out, random_state=random_seed  # type: ignore
    )

    model_keys = list(mapping.keys())
    if model_effects == "mixed":
        for inner_dict in mapping.values():
            model_keys.append(inner_dict["mixed_name"])

    results_all_folds = {}

    ######### OUTER FOLD LOOP #########
    for k, (train_index, test_index) in enumerate(
        tqdm(
            cross_val_split_out(X=X, y=y),
            total=n_splits_out,
            desc=" cv",
            position=0,
            leave=False,
        )
    ):
        print()  # for beautiful tqdm progressbar

        # Assign the outer folds data
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]  # type: ignore

        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]  # type: ignore

        cluster_train = groups.iloc[train_index] if groups is not None else None  # type: ignore
        cluster_test = groups.iloc[test_index] if groups is not None else None  # type: ignore

        if scale_out:
            # apply standard scaler but preserve the type pd.DataFrame
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), columns=X_test.columns, index=X_test.index
            )
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        if slopes is not None:
            Z_train_slope = slopes.iloc[train_index]
            Z_test_slope = slopes.iloc[test_index]

            if scale_out:
                scaler = StandardScaler()
                Z_train_slope_scaled = pd.DataFrame(
                    scaler.fit_transform(Z_train_slope),
                    columns=Z_train_slope.columns,
                    index=Z_train_slope.index,
                )
                Z_test_slope_scaled = pd.DataFrame(
                    scaler.transform(Z_test_slope),
                    columns=Z_test_slope.columns,
                    index=Z_test_slope.index,
                )
            else:
                Z_train_slope_scaled = Z_train_slope
                Z_test_slope_scaled = Z_test_slope

            Z_train_slope_scaled["Intercept"] = 1
            cols = Z_train_slope_scaled.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            Z_train_slope_scaled = Z_train_slope_scaled[cols]

            Z_test_slope_scaled["Intercept"] = 1
            cols = Z_test_slope_scaled.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            Z_test_slope_scaled = Z_test_slope_scaled[cols]

            Z_train = Z_train_slope_scaled.to_numpy()
            Z_test = Z_test_slope_scaled.to_numpy()

        else:
            Z_train = np.ones((len(X_train), 1))  # type: ignore
            Z_test = np.ones((len(X_test), 1))  # type: ignore

        ##### DIAGNOSTICS #####
        if diagnostics:
            if model_effects == "mixed":
                to_diagnose = {
                    "effects": 4,
                    "cluster_train": cluster_train,
                    "cluster_test": cluster_test,
                }
            else:
                to_diagnose = {"effects": 0}
            log_diagnostics(
                X_train_scaled, X_test_scaled, y_train, y_test, run, **to_diagnose
            )

        # Loop over all models
        for model_name in mapping.keys():
            logger.info(f"Evaluating {model_name}...")
            model_instance = mapping[model_name]["model"]
            try:
                param_grid = mapping[model_name][dataset_name]["params"]
            except KeyError:
                param_grid = mapping[model_name]["params"]
                
            # get bool in mapping[model_name]["requires_inner_cv"] and negate it
            skip_inner_cv = not mapping[model_name]["requires_inner_cv"]
            requires_formula = mapping[model_name]["requires_formula"]
            
            if mapping[model_name]["allows_n_jobs"]:
                n_jobs_model_dict = {"n_jobs": mapping[model_name]["n_jobs_model"]}  
            else:
                n_jobs_model_dict = {}
            
            if mapping[model_name]["allows_seed"]:
                model_seed = {"random_state": random_seed}
            else:
                model_seed = {}

            # build inner cv folds
            cross_val_split_in = make_cross_val_split(
                method=split_in, groups=cluster_train, n_splits=n_splits_in, random_state=random_seed  # type: ignore
            )

            if skip_inner_cv:
                # Instantiate model directly
                best_model = model_instance(
                    **n_jobs_model_dict, random_state=random_seed
                )
                best_params = best_model.get_params()
                study = None

            else:
                n_trials
                n_jobs_cv_int = mapping[model_name]["n_jobs_cv"]
                if not scale_in:
                    pipe_in = Pipeline(
                        [
                            (
                                "model",
                                model_instance(**n_jobs_model_dict, **model_seed),
                            ),
                        ]
                    )
                else:
                    pipe_in = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                model_instance(**n_jobs_model_dict, **model_seed),
                            ),
                        ]
                    )

                # add "model__" to all keys of the param_grid
                param_grid = {
                    f"model__{key}": value for key, value in param_grid.items()
                }

                neptune_callback = CustomNeptuneCallback(
                    run[f"{model_name}/Optuna/{k}"],
                    # reduce load on neptune, i.e., reduce number of items and plots
                    log_plot_contour=False,
                    log_plot_optimization_history=True,
                    log_plot_edf=False,
                    study_update_freq=10,  # log every 10th trial,
                )

                if n_trials == "mapped":
                    n_trials = mapping[model_name]["n_trials"]
                if not isinstance(n_trials, int):
                    raise ValueError("Invalid value for n_trials.")

                # generate numpy random_state object for seeding the sampler
                random_state = check_random_state(random_seed)
                sampler_seed = random_state.randint(0, np.iinfo("int32").max)

                # Perform inner cross validation
                sampler = optuna.samplers.TPESampler(seed=sampler_seed)
                study = optuna.create_study(sampler=sampler, direction="maximize")

                study_params = {
                    "cross_val_split": cross_val_split_in,
                    "pipe": pipe_in,
                    "params": param_grid,
                    "X": X_train,
                    "y": y_train,
                }

                study.optimize(
                    lambda trial: objective_cv(
                        trial,
                        **study_params,
                        run=run,
                        n_jobs=n_jobs_cv_int,
                        objective_scorer=objective_scorer,
                    ),
                    n_jobs=1,
                    n_trials=n_trials,
                    callbacks=[neptune_callback],
                )

                best_params = study.best_params
                best_params = {
                    key.replace("model__", ""): value
                    for key, value in best_params.items()
                }

            # add random_state to best_params if it is not already in there
            if "random_state" not in best_params:
                best_params = best_params | model_seed

            # add formula to dict if it is required by the model type
            # to_dict can be unpacked in the fit method
            if requires_formula:
                model_formula = {"formula": formula}
            else:
                model_formula = {}
            
            # Fit the best model on the outer fold
            best_model = model_instance(**best_params)
            fit_result = best_model.fit(X_train_scaled, y_train, **model_formula)

            y_pred = best_model.predict(X_test_scaled)
            y_pred_train = best_model.predict(X_train_scaled)

            all_models_dict = log_single_model_single_fold(
                y_test,
                y_pred,
                y_train,
                y_pred_train,
                model_name,
                best_model,
                best_params,
                k,
                run,
                results_all_folds,
                study=study,
                metrics=metrics,
            )

            # store the results of the outer fold of the current model in a dataclass
            # this makes passing to the postprocessor easier
            single_model_fold_result = SingleModelFoldResult(
                k=k,
                model_name=model_name,
                best_model=best_model,
                best_params=best_params,
                y_pred=y_pred,
                y_test=y_test,
                X_test=X_test_scaled,
                y_pred_train=y_pred_train,
                y_train=y_train,
                X_train=X_train_scaled,
                fit_result=fit_result,
            )

            # call model postprocessing on the single results dataclass
            postpro_func = mapping[model_name]["post_processor"]
            all_models_dict = postpro_func(
                results_all_folds,
                single_model_fold_result,
                run,
                features=X_train.columns,
            )

            ###### MIXED EFFECTS EVALUATION #################
            
            # code that is run for effects == "mixed"
            # if effects == "mixed":
            # look up mixed effects model in mapping
            # get the model instance and load with the base estimator from the level 3 model
            # fit the mixed effects model
            # predict with the mixed effects model
            # store the mixed effects model in the all_models_dict
            # store the mixed effects model in the model_results_dict
            # run the mixed effects model postprocessing
            
            if (model_effects == "mixed") and mapping[model_name]["mixed_name"]:
                
                logger.info(f"Evaluating {mapping[model_name]['mixed_name']}...")
                # tag the base prediction
                y_pred_base = y_pred.copy()
                    
                if mapping[model_name]["mixed_model"] == LinearMixedEffectsModel:
                    # fit the mixed model. For LMEM there is no best model
                    mixed_model_instance = mapping[model_name]["mixed_model"]()
                    # fit the model using the cluster variable and re_formula for slopes 
                    fit_result = mixed_model_instance.fit(
                        X=X_train_scaled,
                        y=y_train,
                        clusters=cluster_train,
                        formula=formula,
                        re_formula=re_formula,
                    )
                elif mapping[model_name]["mixed_model"] == MERF:
                    # instantiate the mixed model with the best fixed effects model
                    mixed_model_instance = mapping[model_name]["mixed_model"](
                        fixed_effects_model=mapping[model_name]["model"](**best_params),
                        max_iterations=max_iterations,
                        gll_early_stop_threshold=em_stopping_threshold,
                        gll_early_stopping_window=em_window,
                        log_gll_per_iteration=False,
                    )
                    # fit the mixed model using cluster variable and Z for slopes
                    fit_result = mixed_model_instance.fit(
                        X=X_train_scaled,
                        y=y_train,
                        clusters=cluster_train,
                        Z=Z_train,
                    )
                else:
                    raise ValueError(f"Invalid mixed model: mixed model must be MERF or LMM but was {mapping[model_name]['mixed_model']}")

                # get test predictions
                y_pred = mixed_model_instance.predict(
                    X=X_test_scaled,
                    clusters=cluster_test,
                    Z=Z_test,
                    predict_known_groups_lmm=predict_known_groups_lmm,
                )
                # get training predictions 
                y_pred_train = mixed_model_instance.predict(
                    X=X_train_scaled,
                    clusters=cluster_train,
                    Z=Z_train,
                    predict_known_groups_lmm=predict_known_groups_lmm,
                )

                all_models_dict = log_single_model_single_fold(
                    y_test,
                    y_pred,
                    y_train,
                    y_pred_train,
                    mapping[model_name]["mixed_name"],
                    best_model,
                    best_params,
                    k,
                    run,
                    results_all_folds,
                    study=None,
                    metrics=metrics,
                )

                single_model_fold_result = SingleModelFoldResult(
                    k=k,
                    model_name=mapping[model_name]["mixed_name"],
                    best_model=best_model,
                    best_params=best_params,
                    y_pred=y_pred,
                    y_test=y_test,
                    X_test=X_test_scaled,
                    y_pred_train=y_pred_train,
                    y_train=y_train,
                    X_train=X_train_scaled,
                    fit_result=fit_result,
                )

                postpro_func = mapping[model_name]["mixed_post_processor"]

                all_models_dict = postpro_func(
                    results_all_folds=all_models_dict,
                    fold_result=single_model_fold_result,
                    run=run,
                    y_pred_base=y_pred_base,
                    mixed_name=mapping[model_name]["mixed_name"],
                )

        if break_cross_val and k == 0:
            break

        print()
        print()

    return results_all_folds


if __name__ == "__main__":
    pass
