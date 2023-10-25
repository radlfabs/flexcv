import numpy as np
import pandas as pd


def select_random_columns(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Select n random columns from a pandas DataFrame and return a new DataFrame containing only these columns.
    :param df: input DataFrame
    :param n: number of columns to select
    :return: DataFrame containing n randomly selected columns
    """
    column_names = df.columns.tolist()
    random_column_names = np.random.choice(column_names, size=n, replace=False)
    return df[random_column_names]


def generate_regression(
    m_features: int, n_samples: int, n_groups: int = 5, n_slopes=1, noise=0.1
):
    """
    Generate a dataset for linear regression using the numpy default rng.
    :param m_features: number of features
    :param n_samples: number of samples
    :param n_groups: number of groups
    :param n_slopes: number of slopes
    :param noise: noise level
    :return: dataset
    """
    FIXED_LEVEL = 0.01
    RANDOM_LEVEL = 1

    # initialize random number generator and generate predictors randomly
    rng = np.random.default_rng(42)
    X = rng.integers(-10, 10, (n_samples, m_features))

    # generate random coefficients with a linear relationship and add noise
    beta_fixed = rng.random(m_features) * FIXED_LEVEL
    epsilon = rng.standard_normal(n_samples) * noise
    # y = 1 + X @ beta + epsilon

    # generate random group labels
    group = rng.integers(0, n_groups, n_samples)

    # generate random effects -> linear relationships for each group
    beta_random = rng.uniform(0.9, 1.0, size=n_groups) * RANDOM_LEVEL
    # reorder the random effects betas according to the group labels
    random_effects = beta_random[group]

    # generate the response variable
    y = 1 + X @ beta_fixed + random_effects + epsilon

    # convert to pandas dataframe with random column names
    X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="y")
    group = pd.Series(group, name="group")

    # select a random column to be the random slope
    random_slopes = select_random_columns(X, n_slopes)

    return X, y, group, random_slopes
