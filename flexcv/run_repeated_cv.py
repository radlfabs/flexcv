import logging
from pprint import pformat

import neptune
import numpy as np
import openpyxl
import pandas as pd
from data_loader import DataLoadConfig
from funcs import add_module_handlers, make_acronym
from main import main
from neptune.types import File
from run import RunConfigurator

from . import cross_val_split as cross_val_split

logger = logging.getLogger(__name__)

api_dict = {
    "api_token": "ANONYMOUS",
    "project": "MYPROJECT",
}

# make a personal_run.py module with this template
# exclude it from git


def aggregate_(repeated_runs) -> pd.DataFrame:
    """Aggregate the results of repeated runs into a single DataFrame.
    Therefore, the nested dict structure of the results is flattened.
    First, the model results are averaged over folds of individual runs.
    Second, the repeated (indicidual) runs are averaged.
    The summary statistics are returned as a DataFrame with the following structure:
    index: [aggregate]_[metric_name]
    columns: [model_name]
    values: [metric_value]
    """

    def try_mean(x):
        try:
            return np.mean(x)
        except ValueError:
            # entered if x contains str "NaN"
            # check if all elements in x are equal to "NaN"
            if np.all([element == "NaN" for element in x]):
                return -99
            else:
                return -999

    model_keys = list(repeated_runs[0].keys())

    result_keys = repeated_runs[0][model_keys[0]]["metrics"].keys()
    results = []
    for result_key in result_keys:
        tmp_df = pd.DataFrame(
            [
                pd.Series(
                    [
                        try_mean(run[model_key]["metrics"][result_key])
                        for model_key in model_keys
                    ],
                    index=model_keys,
                )
                for run in repeated_runs
            ]
        ).agg(["mean", "std"])
        tmp_df.index = [f"{result_key}_{index}" for index in tmp_df.index]
        results.append(tmp_df)
    return pd.concat(results)


if __name__ == "__main__":
    add_module_handlers(logger)

    n_repeats = 3

    dataset = "HSDD"  # , "ISD"]
    target = "CFAPleasantness"  # "CFAEventfulness"  # , ]
    model_level = 3
    custom = cross_val_split.CrossValMethod.CUSTOM
    strat = cross_val_split.CrossValMethod.STRAT
    gkf = cross_val_split.CrossValMethod.GROUPKFOLD
    cv_out, cv_in = (custom, gkf)  # , (gkf, gkf))

    run_config = RunConfigurator(
        DataLoadConfiguration=DataLoadConfig(
            dataset_name=dataset,
            model_level=model_level,
            target_name=target,
            slopes="LAeq",
        ),
        cross_val_method=cv_out,
        cross_val_method_in=cv_in,
        neptune_on=True,
        scale_in=True,
        scale_out=True,
        em_max_iterations=2,
        em_stopping_threshold="mapped",
        em_stopping_window="mapped",
        diagnostics=False,
        n_splits=4,
        n_trials=2,
        refit=False,
        model_selection="best",
        predict_known_groups_lmm=True,
        break_cross_val=False,
    )

    repeated_run = neptune.init_run(**api_dict)
    repeated_id = repeated_run["sys/id"].fetch()
    desc = f"Instance of repeated run {repeated_id}."
    acronym = make_acronym(model_level, dataset, target)
    # set numpy seed to 42.
    # If you do not want to reproduce the repeated run, change the seed or remove the line
    np.random.seed(42)
    seeds = np.random.randint(42000, size=n_repeats).tolist()
    # main returns a tuple of run_id and run_metrics
    # so we call main n_repeats times in the list comprehension
    # and unpack the results into two lists of run_ids and run_metrics using zip
    run_ids, run_metrics = zip(
        *[main(run_config, random_seed=seed, run_description=desc) for seed in seeds]
    )
    df = aggregate_(run_metrics)

    # Print the resulting DataFrame
    print(df)
    df.to_excel("repeated_cv.xlsx")
    repeated_id = repeated_run["sys/id"].fetch()
    repeated_run["summary"].upload(File.as_html(df))
    repeated_run["seeds"] = seeds
    repeated_run[
        "sys/description"
    ] = f"Host run for repeated runs with {n_repeats} repeats. run_ids: {run_ids}"
    repeated_run["RelatedRuns"] = ", ".join(run_ids)
    repeated_run["acronym"] = acronym
    repeated_run["files"].upload("model_mapping.py")
    repeated_run.stop()

    # Add the run meta data to the excel file
    workbook = openpyxl.load_workbook("repeated_cv.xlsx")
    worksheet = workbook.create_sheet("meta")
    worksheet.append([f"Model {acronym}\n"])
    worksheet.append([f"Host Run: {repeated_id}\nRepeats:\n"])
    for run_id in run_ids:
        worksheet.append([run_id])
    worksheet.append([f"Seeds: {seeds}"])
    worksheet.append(
        [
            f"Values of -99 indicate that all elements of the metric were set to the string 'NaN'.\nValues of -999 indicate that the metric could not be computed."
        ]
    )
    worksheet.append([pformat(run_config.__dict__)])
    workbook.save("repeated_cv.xlsx")
