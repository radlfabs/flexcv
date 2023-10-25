import logging
from neptune.metadata_containers.run import Run as NeptuneRun


# def log_run_start(RunConfiguration: RunConfigurator, run: NeptuneRun, run_description: str=""):
#     """
#     Log the start of a run to Neptune.
#     Logs RunConfigurator, SelectedCrossValMethod, Seletion, Model_funcs, Flags for GridSearch and optuna, AV and Group.

#     Parameters:
#     RunConfiguration (RunConfigurator): A RunConfigurator object that contains the configuration for the run.
#     run (NeptuneRun): A Neptune run object.

#     Returns:
#     None
#     """
#     if run_description:
#         run["sys/description"] = run_description
#     run["RunConfiguration"] = pformat(RunConfiguration.__repr__)
#     run["CV Out"] = str(RunConfiguration.cross_val_method.value)
#     run["CV In"] = str(RunConfiguration.cross_val_method_in.value)
#     # run["Scale Inner Fold"] = RunConfiguration.scale_in
#     # run["Scale Outer Fold"] = RunConfiguration.scale_out
#     run["Dataset"] = str(RunConfiguration.DataLoadConfiguration.dataset_name)
#     run["Model"] = str(RunConfiguration.DataLoadConfiguration.model_level)
#     run["Model Selection"] = str(RunConfiguration.model_selection)
#     run["ModelMapping"].upload("model_mapping.py")
#     # run["Optuna"] = RunConfiguration.optuna
#     run["AV"] = RunConfiguration.DataLoadConfiguration.target_name
#     run["files/ModelMapping"].upload("model_mapping.py")
#     run["files/datasets"].upload("datasets.py")
#     run["files/main"].upload("main.py")
#     run["files/cross_val"].upload("cross_val.py")
#     try:
#         run["files/scorer"].upload("scorer.py")
#     except FileNotFoundError:
#         logger.info("scorer.py not found. Skipping upload to Neptune...")

#     run["Acronym"] = make_acronym(
#         model_level=str(RunConfiguration.DataLoadConfiguration.model_level),
#         dataset_name=str(RunConfiguration.DataLoadConfiguration.dataset_name),
#         target_name=str(RunConfiguration.DataLoadConfiguration.target_name),
#     )


# def log_run_end(
#     RunConfiguration: RunConfigurator,
#     run: NeptuneRun,
#     metrics: dict,
#     formula: str,
#     re_formula: str | None,
#     group_name: str,
#     random_slopes: pd.Series | pd.DataFrame | None,
#     df: pd.DataFrame,
# ) -> dict | None:
#     """
#     Log the results of a run in Neptune or return a dictionary of metrics if loop_run is True.

#     Parameters:
#     RunConfiguration (RunConfigurator): Instance of the RunConfigurator class to get information about the run configuration.
#     logger (logging.Logger): Logger instance to log information about the run.
#     run (NeptuneRun): Instance of the NeptuneRun class to log the results in Neptune.
#     metrics (dict): Dictionary of metrics for each estimator in the run.
#     formula (str): Formula used for the run.

#     Returns:
#     dict | None: Returns a dictionary of metrics if loop_run is True, otherwise returns None if loop_run is False.
#     """


#     run["Group"] = group_name

#     # random_slopes is either a pandas Series or DataFrame
#     # we need to log the columns or the series name
#     if random_slopes is not None:
#         if isinstance(random_slopes, pd.DataFrame):
#             run["RS"] = str(random_slopes.columns.tolist())
#         else:
#             run["RS"] = random_slopes.name
#     else:
#         run["RS"] = random_slopes

#     run["formula"] = formula
#     run["re_formula"] = re_formula
#     run["em_max_iterations"] = RunConfiguration.em_max_iterations
#     # try:
#     #     run["files/logfile"].upload("file.log")
#     # except FileNotFoundError:
#     #     logger.info("file.log not found. Skipping upload to Neptune...")
#     run_id = run["sys/id"].fetch()
#     run.stop()

#     return metrics


class Run(NeptuneRun):
    def __init__(self, *args, **kwargs):
        self.run_id = "dummy"

    def fetch(self):
        return self.run_id

    def stop(self):
        pass

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass

    def __getattr__(self, key):
        return self

    def __setattr__(self, key, value):
        pass

    def __delattr__(self, key):
        pass

    def __str__(self):
        return self.run_id

    def __repr__(self):
        return self.run_id

    def append(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def upload(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    for dataset_name in ["HSDD", "ARAUSD", "ISD"]:
        for model_level in range(1, 5):
            for target_name in [
                "CFAPleasantness",
                "CFAEventfulness",
                "ISOPleasantness",
                "ISOEventfulness",
                "Appropriate",
            ]:
                print(make_acronym(model_level, dataset_name, target_name))
