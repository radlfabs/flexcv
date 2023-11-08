import logging
from functools import wraps
import pandas as pd


logger = logging.getLogger(__name__)


def empty_func(*args, **kwargs) -> None:
    """A function that does nothing.

    Args:
      *args: Any argument is accepted.
      **kwargs: Any keayword argument is accepted.

    Returns:
      args, kwargs: The passed arguments and keyword arguments.
    """
    return args, kwargs


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
    """Returns the fixed effects formula for the dataset
    Scheme: "target ~ column1 + column2 + ...

    Args:
      target_name: str: The name of the target variable in the dataset.
      X_data: pd.DataFrame: The feature matrix.

    Returns:
      (str): The fixed effects formula.
    """
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
