import logging
from functools import wraps
import pandas as pd
import neptune


logger = logging.getLogger(__name__)


def add_model_to_keys(param_grid):
    """This function adds the string "model__" to avery key of the param_grid dict.
    
    Args:
      param_grid (dict): A dictionary of parameters for a model.
      
    Returns:
      (dict): A dictionary of parameters for a model with the string "model__" added to each key.
    """
    return {f"model__{key}": value for key, value in param_grid.items()}


def rm_model_from_keys(param_grid):
    """This function removes the string "model__" from avery key of the param_grid dict.
    
    Args:
      param_grid (dict): A dictionary of parameters for a model.
      
    Returns:
      (dict): A dictionary of parameters for a model with the string "model__" removed from each key.
    """
    return {key.replace("model__", ""): value for key, value in param_grid.items()}


def add_module_handlers(logger: logging.Logger) -> None:
    """Adds handlers to the logger for the module.

    Args:
      logger: logging.Logger: The logger for the module.

    Returns:
      (None)
    """
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.INFO)

    # Create a custom logger for the package you want to suppress
    rpy2_logger = logging.getLogger("rpy2")
    rpy2_logger.setLevel(logging.ERROR)  # Suppress log messages for this package

    c_handler = logging.StreamHandler()
    c_format = logging.Formatter("%(module)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)


def get_fixed_effects_formula(target_name, X_data) -> str:
    """Returns the fixed effects formula for the dataset.
    
    Scheme: "target ~ column1 + column2 + ...

    Args:
      target_name: str: The name of the target variable in the dataset.
      X_data: pd.DataFrame: The feature matrix.

    Returns:
      (str): The fixed effects formula.
    """
    if X_data.shape[1] == 1:
        return f"{target_name} ~ {X_data.columns[0]}"
    start = f"{target_name} ~ {X_data.columns[0]} + "
    end = " + ".join(X_data.columns[1:])
    return start + end


def get_re_formula(random_slopes_data):
    """Returns a random effects formula for use in statsmodels. Scheme: ~ random_slope1 + random_slope2 + ...

    Args:
      random_slopes_data: pd.Series | pd.DataFrame: The random slopes data.

    Returns:
      (str): The random effects formula.
    """
    if random_slopes_data is None:
        return ""
    elif isinstance(random_slopes_data, pd.DataFrame):
        return "~ " + " + ".join(random_slopes_data.columns)
    elif isinstance(random_slopes_data, pd.Series):
        return "~ " + random_slopes_data.name
    else:
        raise TypeError("Random slopes data type not recognized")


def run_padding(func):
    """Decorator to add padding to the output of a function.
    Helps to visually separate the output of different functions.

    Args:
      func: Any callable.

    Returns:
      (Any): Return value of the passed callable.
    """

    @wraps(func)
    def wrapper_function(*args, **kwargs):
        print()
        print("~" * 10, "STARTING RUN", "~" * 10)
        print()
        results = func(*args, **kwargs)
        print()
        print("~" * 10, " END OF RUN", "~" * 10)
        print()
        return results

    return wrapper_function


def pformat_dict(d, indent=""):
    """Pretty-format a dictionary, only printing values that are themselves dictionaries.

    Args:
      d (dict): dictionary to print
      indent (str): Level of indentation for use with recursion (Default value = "")

    Returns:

    """
    formatted = ""
    for key, value in d.items():
        formatted.join(f"{indent}{key}")
        if isinstance(value, dict):
            next_layer = pformat_dict(value, indent + "  ")
            formatted.join(next_layer)
    return formatted


def get_repeated_cv_metadata(str_children="Instance of repeated run ", api_dict=None):
    """This function can be used to fetch metadata from repeated cross-validation runs.
    We use it to get the ids of the children runs and their descriptions.

    Args:
        str_children (str): The string that is prepended to the description of each child run.
        api_dict (dict): A dictionary containing the Neptune.ai project name and the api token.
    """
    if api_dict is None:
        raise ValueError("api_dict must be provided")

    # get a list of all runs in the project
    project = neptune.init_project(
        project=api_dict["project"],
        api_token=api_dict["api_token"],
        mode="read-only",
    )
    runs_table_df = project.fetch_runs_table().to_pandas()
    # use only rows where "sys/description" begins with "Instance"
    # group by run sys/description
    grouped = runs_table_df[
        runs_table_df["sys/description"].str.startswith(str_children)
    ].groupby("sys/description")
    # get sys/id for each group
    grouped_ids = grouped["sys/id"].apply(list)
    # remove "Instance of repeated run " and trailing dot from the description
    grouped_ids.index = grouped_ids.index.str.replace(str_children, "")
    grouped_ids.index = grouped_ids.index.str.replace(".", "")
    # rename the index to "host id"
    grouped_ids.index.name = "host id"
    # rename the column to "children ids"
    grouped_ids.name = "children ids"
    metadata = pd.DataFrame(grouped_ids)
    # use the host ids to get their sys/description and make them a new column in the DataFrame
    host_ids = grouped_ids.index
    descriptions = runs_table_df[runs_table_df["sys/id"].isin(host_ids)][
        "sys/description"
    ]
    descriptions.index = host_ids
    descriptions.index.name = "host id"
    descriptions.name = "description"
    # join the two DataFrames
    metadata = metadata.join(pd.DataFrame(descriptions))
    # save to excel
    metadata.to_excel("repeated_cv_metadata.xlsx")