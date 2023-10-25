import logging
from functools import wraps
import os

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def add_module_handlers(logger: logging.Logger):
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


def get_fixed_effects_formula(target_name, X_data):
    """Returns the fixed effects formula for the dataset
    Scheme: "target ~ column1 + column2 + ..."""
    start = f"{target_name} ~ {X_data.columns[0]} + "
    end = " + ".join(X_data.columns[1:])
    return start + end


def get_re_formula(random_slopes_data):
    """Scheme: ~ random_slope1 + random_slope2 + ..."""
    if random_slopes_data is None:
        return ""
    elif isinstance(random_slopes_data, pd.DataFrame):
        return "~ " + " + ".join(random_slopes_data.columns)
    elif isinstance(random_slopes_data, pd.Series):
        return "~ " + random_slopes_data.name
    else:
        raise TypeError("Random slopes data type not recognized")


def run_padding(func):
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
