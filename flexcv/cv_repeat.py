import logging

import numpy as np
import pandas as pd
from neptune.types import File

from .cv_class import CrossValidation
from .funcs import add_module_handlers
from .run import Run

logger = logging.getLogger(__name__)


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


def perform_repeats(cv: CrossValidation, n_repeats=3):
    add_module_handlers(logger)
    repeated_run = Run()  # neptune.init_run(**your_credentials)
    repeated_id = repeated_run["sys/id"].fetch()
    desc = f"Instance of repeated run {repeated_id}."

    # set numpy seed to 42.
    # If you do not want to reproduce the repeated run, change the seed or remove the line
    np.random.seed(42)

    # generate a list of random seeds to use in the repeated loop
    # the random seeds are used to generate the random folds in the repeated loop
    seeds = np.random.randint(42000, size=n_repeats).tolist()

    run_ids = []
    run_results = []
    for seed in seeds:
        # create a new run for each repeat
        # neptune will log every inner run as well as the host run
        inner_run = Run()  # neptune.init_run(**your_credentials)
        inner_id = inner_run["sys/id"].fetch()
        inner_run["HostRun"] = repeated_id
        inner_run["seed"] = seed

        results = (
            cv
            .set_run(run=inner_run, random_seed=seed)
            .perform()
            .get_results()
        )

        # append the run id and the run metric to the lists
        run_ids.append(inner_id)
        run_results.append(results)

    # run_dfs have the same column and index names and we
    df = aggregate_(run_results)
    df.to_excel("repeated_cv.xlsx")  # save dataframe to excel file
    print(df)  # print dataframe to console

    # log the repeated run results to neptune
    repeated_id = repeated_run["sys/id"].fetch()
    repeated_run["summary"].upload(File.as_html(df))
    repeated_run[
        "sys/description"
    ] = f"Host run for repeated runs with {n_repeats} repeats. run_ids: {run_ids}"
    repeated_run["RelatedRuns"] = ", ".join(run_ids)
    repeated_run["seeds"] = seeds
    repeated_run["mapping"] = cv.config["mapping"]
    repeated_run.stop()


if __name__ == "__main__":
    from flexcv import CrossValidation
    from flexcv.data_generation import generate_regression
    from flexcv.models import LinearModel
    from flexcv.model_mapping import ModelConfigDict, ModelMappingDict

    # make sample data
    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)

    # create a model mapping
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                }
            ),
        }
    )

    # instantiate our cross validation class
    cv = CrossValidation().set_data(X, y, group, dataset_name="ExampleData")

    # call perform_repeats function
    perform_repeats(cv, 3)
