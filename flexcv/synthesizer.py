import numpy as np
import pandas as pd


def select_random_columns(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select n random columns from a pandas DataFrame and return a new DataFrame containing only these columns.

    Args:
      df: pd.DataFrame: Input data to select n columns from.
      n: int: Number of columns to select.

    Returns:
      : pd.DataFrame: A DataFrame containing n randomly selected columns

    """
    column_names = df.columns.tolist()
    random_column_names = np.random.choice(column_names, size=n, replace=False)
    return df[random_column_names]


def generate_regression(
    m_features: int, n_samples: int, n_groups: int=5, n_slopes: int=1, noise_level: float=0.1
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Generate a dataset for linear regression using the numpy default rng.

    Args:
      m_features: int: Number of features, i.e. columns, to be generated.
      n_samples: int: Number of rows to be generated.
      n_groups: int: Number of groups/clusters. (Default value = 5)
      n_slopes: int: Number of columns in the feature matrix to be used as random slopes as well. (Default value = 1)
      noise_level: float: The data will be generated with added standard normal noise which is multiplied with noise_level. (Default value = 0.1)

    Returns:
      : tuple: A tuple containing the following elements:
        (The feature matrix DataFrame, the target vector Series, the group labels Series, the random slopes DataFrame)

    """
    FIXED_LEVEL = 0.01
    RANDOM_LEVEL = 1

    # initialize random number generator and generate predictors randomly
    rng = np.random.default_rng(42)
    X = rng.integers(-10, 10, (n_samples, m_features))

    # generate random coefficients with a linear relationship and add noise
    beta_fixed = rng.random(m_features) * FIXED_LEVEL
    epsilon = rng.standard_normal(n_samples) * noise_level
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
