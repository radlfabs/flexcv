import numpy as np
import pandas as pd


def generate_regression(
    m_features: int,
    n_samples: int,
    n_groups: int = 5,
    n_slopes: int = 1,
    random_seed: int = 42,
    noise_level: float = 0.1,
    fixed_random_ratio: float = 0.01,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Generate a dataset for linear regression using the numpy default rng.

    Args:
      m_features (int): Number of features, i.e. columns, to be generated.
      n_samples (int): Number of rows to be generated.
      n_groups (int): Number of groups/clusters. (Default value = 5)
      n_slopes (int): Number of columns in the feature matrix to be used as random slopes as well. (Default value = 1)
      noise_level (float): The data will be generated with added standard normal noise which is multiplied with noise_level. (Default value = 0.1)
      fixed_random_ratio (float): The ratio of the fixed effects to the random effects. (Default value = 0.01)
      random_seed (int): The random seed to be used for reproducibility. (Default value = 42)
      
    Returns:
      (tuple): A tuple containing the following elements:
                (The feature matrix DataFrame, the target vector Series, the group labels Series, the random slopes DataFrame)

    """

    # initialize random number generator and generate predictors randomly
    rng = np.random.default_rng(random_seed)
    X = rng.integers(-10, 10, (n_samples, m_features))

    # generate random coefficients with a linear relationship and add noise
    beta_fixed = rng.random(m_features) * fixed_random_ratio
    epsilon = rng.standard_normal(n_samples) * noise_level

    # generate random group labels
    group = rng.integers(0, n_groups, n_samples)

    # generate random effects -> linear relationships for each group
    beta_random = rng.uniform(0.9, 1.0, size=n_groups)
    # reorder the random effects betas according to the group labels
    random_effects = beta_random[group]

    # generate the response variable
    y = 1 + X @ beta_fixed + random_effects + epsilon

    # convert to pandas dataframe with random column names
    X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="y")
    group = pd.Series(group, name="group")

    # select a random column to be the random slope
    column_names = X.columns.tolist()
    choosen_columns = rng.choice(column_names, size=n_slopes, replace=False)
    random_slopes = X[choosen_columns]

    return X, y, group, random_slopes
