import logging
import warnings
from typing import Dict, Iterator

import numpy as np
import optuna
import pandas as pd
from neptune.integrations.python_logger import NeptuneHandler
from neptune.metadata_containers.run import Run as NeptuneRun
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_random_state
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.model_selection._split import BaseCrossValidator
from tqdm import tqdm

from .fold_logging import (
    CustomNeptuneCallback,
    log_diagnostics,
)
from .fold_results_handling import SingleModelFoldResult
from .metrics import MetricsDict, mse_wrapper
from .model_selection import ObjectiveScorer, objective_cv
from .split import CrossValMethod, make_cross_val_split
from .utilities import (
    get_fixed_effects_formula, 
    get_re_formula,
    add_model_to_keys,
    rm_model_from_keys,
)
from .model_mapping import ModelMappingDict
from .merf import MERF
from .model_postprocessing import MERFModelPostprocessor
from .utilities import handle_duplicate_kwargs


warnings.filterwarnings("ignore", module=r"matplotlib\..*")
warnings.filterwarnings("ignore", module=r"xgboost\..*")
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)


def preprocess_slopes(Z_train_slope: pd.DataFrame | pd.Series, Z_test_slope: pd.DataFrame | pd.Series, must_scale: bool) -> tuple[np.ndarray, np.ndarray]:
    """This function preprocesses the random slopes variable(s) for use in the mixed effects model.
    
    Args:
        Z_train_slope (pd.DataFrame | pd.Series): Random slopes variable(s) for the training set.
        Z_test_slope (pd.DataFrame | pd.Series): Random slopes variable(s) for the test set.
        must_scale (bool): If True, the random slopes are scaled to zero mean and unit variance.
        
    Returns:
        (tuple[np.ndarray, np.ndarray]): The preprocessed random slopes as a tuple of numpy arrays: (Z_train, Z_test)
    """
    is_dataframe_train = isinstance(Z_train_slope, pd.DataFrame)
    is_dataframe_test = isinstance(Z_test_slope, pd.DataFrame)
    if not is_dataframe_train and not isinstance(Z_train_slope, pd.Series):
        raise TypeError(
            f"Z_train_slope must be a pandas DataFrame or pandas Series, not {type(Z_train_slope)}"
        )
    if not is_dataframe_test and not isinstance(Z_test_slope, pd.Series):
        raise TypeError(
            f"Z_test_slope must be a pandas DataFrame or pandas Series, not {type(Z_test_slope)}"
        )
    if not isinstance(must_scale, bool):
        raise TypeError(
            f"must_scale must be a bool, not {type(must_scale)}"
        )
    
    # check dimensions
    if is_dataframe_train and (Z_train_slope.shape[1] != Z_test_slope.shape[1]):
        raise ValueError(
            f"Z_train_slope and Z_test_slope must have the same number of columns. Z_train_slope has {Z_train_slope.shape[1]} columns, Z_test_slope has {Z_test_slope.shape[1]} columns."
        )
    # convert to DataFrame
    if not is_dataframe_train:
        Z_train_slope = pd.DataFrame(Z_train_slope)
    if not is_dataframe_test:
        Z_test_slope = pd.DataFrame(Z_test_slope)
    
    if must_scale:
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
    return Z_train, Z_test


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scales the features to zero mean and unit variance.
    
    Args:
        X_train (pd.DataFrame): Features for the training set.
        X_test (pd.DataFrame): Features for the test set.
        
    Returns:
        (tuple[pd.DataFrame, pd.DataFrame]): The preprocessed features as a tuple of pandas DataFrames: (X_train_scaled, X_test_scaled)
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            f"X_train must be a pandas DataFrame, not {type(X_train)}"
        )
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"X_test must be a pandas DataFrame, not {type(X_test)}"
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of columns. X_train has {X_train.shape[1]} columns, X_test has {X_test.shape[1]} columns."
        )
    
    scaler = StandardScaler()
        
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled
    

def cross_validate(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    run: NeptuneRun,
    groups: pd.Series,
    slopes: pd.DataFrame | pd.Series,
    split_out: CrossValMethod | BaseCrossValidator | Iterator,
    split_in: CrossValMethod | BaseCrossValidator | Iterator,
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
    em_max_iterations: int = None,
    em_stopping_threshold: float = None,
    em_stopping_window: int = None,
    predict_known_groups_lmm: bool = True,
    diagnostics: bool = False,
    **kwargs
) -> Dict[str, Dict[str, list]]:
    """This function performs a cross-validation for a given regression formula, using one or a number of specified machine learning models and a configurable cross-validation method.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        target_name (str): Custom target name.
        run (NeptuneRun): A Run object to log to.
        groups (pd.Series): The grouping or clustering variable.
        slopes (pd.DataFrame | pd.Series): Random slopes variable(s)
        split_out (CrossValMethod | BaseCross): Outer split strategy.
        split_in (CrossValMethod): Inner split strategy.
        break_cross_val (bool): If True, only the first outer fold is evaluated.
        scale_in (bool): If True, the features are scaled in the inner cross-validation to zero mean and unit variance. This works independently of the outer scaling.
        scale_out (bool): If True, the features are scaled in the outer cross-validation to zero mean and unit variance.
        n_splits_out (int): Number of outer cross-validation folds.
        n_splits_in (int): Number of inner cross-validation folds.
        random_seed (int): Seed for all random number generators.
        model_effects (str): If "fixed", only fixed effects are used. If "mixed", both fixed and random effects are used.
        n_trials (int): Number of trials for the inner cross-validation, i.e. the number of hyperparameter combinations to sample from the distributions.
        mapping (ModelMappingDict): The mapping providing model instances, hyperparameter distributions, and postprocessing functions.
        metrics (MetricsDict): A dict of metrics to be used as the evaluation metric for the outer cross-validation.
        objective_scorer (ObjectiveScorer): A custom objective scorer object to provide the evaluation metric for the inner cross-validation.
        em_max_iterations (int): For use with MERF. Maximum number of iterations for the EM algorithm. (Default: None)
        em_stopping_threshold (float): For use with MERF. Threshold for the early stopping criterion of the EM algorithm. (Default: None)
        em_stopping_window (int): For use with MERF. Window size for the early stopping criterion of the EM algorithm. (Default: None)
        predict_known_groups_lmm (bool): For use with Mixed Linear Models. If True, the model will predict the known groups in the test set. (Default: True)
        diagnostics (bool): If True, diagnostics plots are logged to Neptune. (Default: False)
        **kwargs: Additional keyword arguments.


    Returns:
      Dict[str, Dict[str, list]]: A dictionary containing the results of the cross-validation, organized by machine learning models.

    The function returns a nested dictionary with the following structure:
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

    print()
    re_formula = get_re_formula(slopes)
    formula = get_fixed_effects_formula(target_name, X)
    cross_val_split_out = make_cross_val_split(
        method=split_out, groups=groups, n_splits=n_splits_out, random_state=random_seed  # type: ignore
    )

    model_keys = list(mapping.keys())
    for model_name, inner_dict in mapping.items():
        if "add_merf" in inner_dict and inner_dict["add_merf"]:
            merf_name = f"{model_name}_MERF"
            model_keys.append(merf_name)

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
        
        groups_exist = groups is not None
        slopes_exist = slopes is not None 
        
        # check if break is requested and if this is the 2nd outer fold
        if break_cross_val and k == 1:
            break
        
        # Assign the outer folds data
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]  # type: ignore

        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]  # type: ignore

        cluster_train = groups.iloc[train_index] if groups_exist else None  # type: ignore
        cluster_test = groups.iloc[test_index] if groups_exist else None  # type: ignore

        if scale_out:
            X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        if slopes_exist:
            Z_train, Z_test = preprocess_slopes(
                Z_train_slope=slopes.iloc[train_index], 
                Z_test_slope=slopes.iloc[test_index], 
                must_scale=scale_out
            )

        else:
            Z_train = np.ones((len(X_train), 1))  # type: ignore
            Z_test = np.ones((len(X_test), 1))  # type: ignore

        # run diagnostics if requested
        if diagnostics:
            to_diagnose = {
                "effects": model_effects,
                "cluster_train": cluster_train,
                "cluster_test": cluster_test,
            } if model_effects == "mixed" else {"effects": model_effects}

            log_diagnostics(
                X_train_scaled, X_test_scaled, y_train, y_test, run, **to_diagnose
            )

        # Loop over all models
        for model_name in mapping.keys():
            logger.info(f"Evaluating {model_name}...")
        
            skip_inner_cv = not mapping[model_name]["requires_inner_cv"]  # get bool in mapping[model_name]["requires_inner_cv"] and negate it
            evaluate_merf = mapping[model_name]["add_merf"]

            model_class = mapping[model_name]["model"]
            param_grid = mapping[model_name]["params"]
            model_kwargs = mapping[model_name]["model_kwargs"]
            
            # build inner cv folds
            cross_val_split_in = make_cross_val_split(
                method=split_in, groups=cluster_train, n_splits=n_splits_in, random_state=random_seed  # type: ignore
            )

            if skip_inner_cv:
                # Instantiate model directly without inner cross-validation
                # has to be best_model because it is the only one instantiated
                best_model = model_class(
                    **model_kwargs
                )
                best_params = best_model.get_params()
                # set study to None since no study is instantiated otherwise
                study = None

            else:
                # this block performs the inner cross-validation with Optuna
                
                n_trials = mapping[model_name]["n_trials"]
                n_jobs_cv_int = mapping[model_name]["n_jobs_cv"]

                pipe_in = Pipeline(
                    [
                        ("scaler", StandardScaler()) if scale_in else (),
                        (
                            "model",
                            model_class(**model_kwargs),
                        ),
                    ]
                )

                # add "model__" to all keys of the param_grid
                param_grid = add_model_to_keys(param_grid)
                
                neptune_callback = CustomNeptuneCallback(
                    run[f"{model_name}/Optuna/{k}"],
                    # reduce load on neptune, i.e., reduce number of items and plots
                    log_plot_contour=False,
                    log_plot_optimization_history=True,
                    log_plot_edf=False,
                    study_update_freq=10,  # log every 10th trial,
                )

                # generate numpy random_state object for seeding the sampler
                random_state = check_random_state(random_seed)
                sampler_seed = random_state.randint(0, np.iinfo("int32").max)

                # get sampler to be used for the inner cross-validation
                sampler = optuna.samplers.TPESampler(seed=sampler_seed)
                
                # instantiate the study object
                study = optuna.create_study(sampler=sampler, direction="maximize")

                # run the inner cross-validation
                study.optimize(
                    lambda trial: objective_cv(
                        trial,
                        cross_val_split=cross_val_split_in,
                        pipe=pipe_in,
                        params=param_grid,
                        X=X_train,
                        y=y_train,
                        run=run,
                        n_jobs=n_jobs_cv_int,
                        objective_scorer=objective_scorer,
                    ),
                    n_jobs=1,
                    n_trials=n_trials,
                    callbacks=[neptune_callback],
                )
                # get best params from study and rm "model__" from keys
                best_params = rm_model_from_keys(study.best_params)

            # add random_state to best_params if it is not already in there
            if "random_state" not in best_params and "random_state" in model_kwargs:
                best_params.update({"random_state": random_seed})

            # add formula to dict if it is required by the model type
            # to_dict can be unpacked in the fit method

            train_pred_kwargs = {}
            test_pred_kwargs = {}
            pred_kwargs = {}
            
            if mapping[model_name]["consumes_clusters"]:
                model_kwargs["clusters"] = cluster_train
                model_kwargs["re_formula"] = re_formula
                model_kwargs["formula"] = formula
                
                pred_kwargs["predict_known_groups_lmm"] = predict_known_groups_lmm
                
                test_pred_kwargs["clusters"] = cluster_test
                test_pred_kwargs["Z"] = Z_test
                
                train_pred_kwargs["clusters"] = cluster_train
                train_pred_kwargs["Z"] = Z_train
            
            # make new instance of the model with the best parameters
            best_model = model_class(
                **handle_duplicate_kwargs(model_kwargs, best_params)
                )
            
            # Fit the best model on the outer fold
            fit_result = best_model.fit(
                    X=X_train_scaled,
                    y=y_train,
                )
            # get test predictions
            y_pred = best_model.predict(
                X=X_test_scaled,
                **handle_duplicate_kwargs(pred_kwargs, test_pred_kwargs),
            )
            # get training predictions
            y_pred_train = best_model.predict(
                X=X_train_scaled,
                **handle_duplicate_kwargs(pred_kwargs, train_pred_kwargs),
            )
            
            # store the results of the outer fold of the current model in a dataclass
            # this makes passing to the postprocessor easier
            model_data = SingleModelFoldResult(
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
            all_models_dict = model_data.make_results(run=run, results_all_folds=results_all_folds, study=study, metrics=metrics)
            
            try:
                # call model postprocessing on the single results dataclass
                postprocessor = mapping[model_name]["post_processor"]()
                all_models_dict = postprocessor(
                    results_all_folds,
                    model_data,
                    run,
                    features=X_train.columns,
                )
            except KeyError:
                logger.info(
                    f"No postprocessor passed for {model_name}. Moving on..."
                )

            if evaluate_merf:
            ###### MERF EVALUATION #################
            # The base model is passed to the MERF class for evaluation in Expectation Maximization (EM) algorithm

                merf_name = f"{model_name}_MERF"
                logger.info(f"Evaluating {merf_name}...")
                
                # tag the base prediction
                y_pred_base = y_pred.copy()

                # instantiate the mixed model with the best fixed effects model
                merf = MERF(
                    fixed_effects_model=mapping[model_name]["model"](**best_params),
                    max_iterations=em_max_iterations,
                    gll_early_stop_threshold=em_stopping_threshold,
                    gll_early_stopping_window=em_stopping_window,
                    log_gll_per_iteration=False,
                    **model_seed,
                )
                # fit the mixed model using cluster variable and Z for slopes
                fit_result = merf.fit(
                    X=X_train_scaled,
                    y=y_train,
                    clusters=cluster_train,
                    Z=Z_train,
                )

                # fit the model using the cluster variable and re_formula for slopes
                fit_result = merf.fit(
                        X=X_train_scaled,
                        y=y_train,
                        clusters=cluster_train,
                        formula=formula,
                        re_formula=re_formula,
                    )
                # get test predictions
                y_pred = merf.predict(
                    X=X_test_scaled,
                    clusters=cluster_test,
                    Z=Z_test,
                    predict_known_groups_lmm=predict_known_groups_lmm,
                )
                # get training predictions
                y_pred_train = merf.predict(
                    X=X_train_scaled,
                    clusters=cluster_train,
                    Z=Z_train,
                    predict_known_groups_lmm=predict_known_groups_lmm,
                )

                merf_data = SingleModelFoldResult(
                    k=k,
                    model_name=mapping[model_name]["mixed_name"],
                    best_model=merf,
                    best_params=best_params,
                    y_pred=y_pred,
                    y_test=y_test,
                    X_test=X_test_scaled,
                    y_pred_train=y_pred_train,
                    y_train=y_train,
                    X_train=X_train_scaled,
                    fit_result=fit_result,
                )
                
                all_models_dict = merf_data.make_results(study=study, metrics=metrics)
                
                postprocessor = MERFModelPostprocessor()
                all_models_dict = postprocessor(
                    results_all_folds=all_models_dict,
                    fold_result=merf_data,
                    run=run,
                    y_pred_base=y_pred_base,
                    mixed_name=mapping[model_name]["mixed_name"],
                )

        print()
        print()

    return results_all_folds


if __name__ == "__main__":
    pass
