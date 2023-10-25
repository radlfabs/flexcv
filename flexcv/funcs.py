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


def make_acronym(model_level: int, dataset_name: str, target_name: str) -> str:
    MODEL_ACRONYM = {
        "CFAPleasantness": "Pc",
        "CFAEventfulness": "Ec",
        "ISOPleasantness": "Pi",
        "ISOEventfulness": "Ei",
        "Appropriate": "A",
        "Appropriateness": "A",
    }

    return dataset_name[0] + MODEL_ACRONYM[target_name] + str(model_level)


def add_module_handlers(logger: logging.Logger):
    # logger.setLevel(logging.INFO)

    # c_handler = logging.StreamHandler()
    # c_format = logging.Formatter("%(module)s - %(levelname)s - %(message)s")
    # c_handler.setFormatter(c_format)
    # c_handler.setLevel(logging.INFO)
    # logger.addHandler(c_handler)

    # f_handler = logging.FileHandler("file.log")
    # f_format = logging.Formatter(
    #     "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    # )
    # f_handler.setFormatter(f_format)
    # f_handler.setLevel(logging.DEBUG)
    # logger.addHandler(f_handler)
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
    # delete an existing log file if it exists
    # if os.path.exists("file.log"):
    #     # delete all lines in file.log
    #     open("file.log", "w").close()
    # f_handler = logging.FileHandler("file.log")
    # f_format = logging.Formatter(
    #     "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    # )
    # f_handler.setFormatter(f_format)
    # f_handler.setLevel(logging.DEBUG)
    # logger.addHandler(f_handler)


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


def describe_nan(df):
    """Returns a DataFrame with the number of NaNs and the NaN rate for each column."""
    return pd.DataFrame(
        [
            (i, df[df[i].isna()].shape[0], df[df[i].isna()].shape[0] / df.shape[0])
            for i in df.columns
        ],
        columns=["column", "nan_counts", "nan_rate"],
    )


def run_padding(func):
    @wraps(func)
    def wrapper_function(*args, **kwargs):
        print()
        print("~" * 10, "STARTING RUN", "~" * 10)
        # print("Objective:", args, kwargs)
        print()
        results = func(*args, **kwargs)
        print()
        print("~" * 10, " END OF RUN", "~" * 10)
        print()
        return results

    return wrapper_function


def select_cols(df: pd.DataFrame, cols: list):
    return df.loc[:, [column for column in cols if column in df.columns]]


def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        correlations,
        vmax=1.0,
        center=0,
        fmt=".2f",
        cmap="YlGnBu",
        square=True,
        linewidths=0.5,
        annot=True,
        cbar_kws={"shrink": 0.70},
    )
    plt.show()


def get_k_important_shap_features(
    shap_values: np.ndarray, X: pd.DataFrame, k: int = 3
) -> pd.Series:
    """This function takes shap_values and the X DataFrame and return the k most important feature names.

    Args:
        shap_values (np.ndarray): _description_
        X (pd.DataFrame): _description_
        k (int, optional): _description_. Defaults to 3.

    Returns:
        pd.Series: _description_
    """
    return X.columns[np.argsort(np.abs(shap_values).mean(0))][::-1][0:k]


def drop_duplicate_columns(df):
    # Get boolean mask of duplicated columns
    duplicated_cols = np.array(df.columns.duplicated())
    return df.loc[:, ~duplicated_cols]


def max_value_for_file_pairs(df):
    # Group the DataFrame by the filename prefix
    return df.groupby(df.index).max()
