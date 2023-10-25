import logging
import pandas as pd
import warnings
from pprint import pformat

import numpy as np
from neptune.integrations.python_logger import NeptuneHandler
import matplotlib as mpl

mpl.use("Agg")

from sklearn.preprocessing import StandardScaler


from run import RunConfigurator
from run import init_logging
from run import log_run_start
from run import log_run_end
from cross_validate import cross_validate
import flexcv.cv_split as cross_val_split
from funcs import add_module_handlers
from funcs import run_padding

warnings.filterwarnings("ignore", module="matplotlib\..*")
warnings.filterwarnings("ignore", module="xgboost\..*")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)
logging.getLogger("neptune").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# add_module_handlers(logger)

np.random.seed(42)

api_dict = {
    "api_token": "ANONYMOUS",
    "project": "MYPROJECT",
}


@run_padding
def main(RunConfiguration: RunConfigurator, random_seed=42, run_description=""):
    """
    This function performs the main operation of a machine learning pipeline.
    It takes in a RunConfiguration object, which contains all the configurations for the run.
    It initializes logging, which is stored in a file named main.log, and sets the logging level to logging.DEBUG if RunConfiguration.debug is True, or logging.INFO otherwise.
    It also initializes Neptune run and stores various run-related information in the Neptune run object.

    The function then initializes a DataLoader object, which is used to load the data from the dataset.
    The DataLoader object is initialized with the DataLoadConfiguration object from the RunConfiguration object.
    The DataLoader object is then used to load the data from the dataset.
    The data is loaded into three pandas DataFrames: X, y, and group.
    X contains the features, y contains the target variable, and group contains the group variable.
    The DataLoader is then used to generate the fixed effects formula, which is used in the cross validation.

    The function then performs cross validation.
    The cross validation is performed by the cross_validate function, which is imported from cross_val.py.
    The cross_validate function takes in the RunConfiguration object, the X, y, and group DataFrames, and the fixed effects formula.
    The cross_validate function returns a dictionary of metrics, which is stored in the results variable.

    The function then performs refits.
    The refits are performed by the refit_log_median_model function, which is imported from refit.py.
    The refit_log_median_model function takes in the RunConfiguration object, the fixed effects formula, the results dictionary, the estimator, the X, y, and group DataFrames, and the Neptune run object.
    The refit_log_median_model function performs the refit and logs the results to Neptune.

    In the end the function logs the end of the run to Neptune and returns the results dictionary.

    @run_padding is a decorator for the function, which adds Start and End print-statements to the run.

    Arguments:
        RunConfiguration {RunConfigurator} -- RunConfiguration object, which contains all the configurations for the run.
        random_seed {int} -- Random seed to use for the run. Defaults to 42. The seed is used to control reproducibility of random processed in cross validation and model building.
    """
    run = init_logging(RunConfiguration, api_dict)
    npt_handler = NeptuneHandler(run=run)
    add_module_handlers(logger)
    logger.addHandler(npt_handler)
    log_run_start(RunConfiguration, run, run_description)

    data_loader = DataLoader(RunConfiguration.DataLoadConfiguration, n_obs_per_group=40)
    X, y, group, random_slopes = data_loader.get_data()
    data_loader.log(run, "run_dfs/")
    run["groups_value_counts"] = pformat(group.value_counts().to_dict())
    run["n_groups"] = group.nunique()

    formula = data_loader.get_fixed_effects_formula()

    if RunConfiguration.DataLoadConfiguration.slopes:
        re_formula = data_loader.get_re_formula()
    else:
        re_formula = None
        random_slopes = None

    logger.info(f"Number of Columns in Selected Dataset: {X.columns.shape[0]}")
    logger.info(f"Number of Rows in Selected Dataset: {X.shape[0]}")
    logger.info(f"{formula=}")
    logger.info(
        f"Perform outer cross validation method: {RunConfiguration.cross_val_method.value}"
    )
    logger.info(
        f"Perform inner cross validation method: {RunConfiguration.cross_val_method_in.value}"
    )

    results = cross_validate(
        RunConfiguration=RunConfiguration,
        X=X,
        y=y,
        group=group,
        slopes=random_slopes,
        re_formula=re_formula,
        formula=formula,
        run=run,
        random_seed=random_seed,
    )

    run_id = run["sys/id"].fetch()
    results_dict = log_run_end(
        RunConfiguration=RunConfiguration,
        run=run,
        metrics=results,
        formula=formula,
        re_formula=re_formula,
        group_name=group.name,
        random_slopes=random_slopes,
        df=pd.concat((X, y, group)),
    )

    return run_id, results_dict


if __name__ == "__main__":
    dataset = "HSDD"  # , "ISD"]
    target = "CFAEventfulness"  # , "CFAPleasantness"]
    model_level = 4
    custom = cross_val_split.CrossValMethod.CUSTOM
    strat = cross_val_split.CrossValMethod.STRAT
    gkf = cross_val_split.CrossValMethod.GROUPKFOLD
    cv_out, cv_in = (strat, gkf)  # , (gkf, gkf))

    main(
        RunConfigurator(
            DataLoadConfiguration=DataLoadConfig(
                dataset_name=dataset,
                model_level=model_level,
                target_name=target,
                slopes="LAeq",  # TODO enable slopes here
            ),
            cross_val_method=cv_out,
            cross_val_method_in=cv_in,
            neptune_on=True,
            scale_in=True,
            scale_out=True,
            em_max_iterations="mapped",
            em_stopping_threshold="mapped",
            em_stopping_window="mapped",
            diagnostics=False,
            n_splits=5,
            n_trials="mapped",
            refit=False,
            model_selection="best",
            predict_known_groups_lmm=True,
            break_cross_val=False,
        )
    )
