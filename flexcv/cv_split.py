from enum import Enum
from functools import partial
from typing import Callable, Iterator

import pandas as pd
from numpy import ndarray
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import KBinsDiscretizer


class CrossValMethod(Enum):
    """Enum class to assign CrossValMethods to the cross_val() function.
    This is useful to return the correct splitting function depending on the cross val method.
    Implemented methods are:
    KFOLD: KFold cross validation
    CUSTOMSTRAT: CustomStratifiedKFold cross validation
    GROUP: GroupKFold cross validation
    STRATGROUP: StratifiedGroupKFold cross validation
    CUSTOMSTRATGROUP: CustomStratifiedGroupKFold cross validation
    
    Details:
    - KFOLD: Regular sklearn KFold cross validation. No grouping information is used.
    - CUSTOMSTRAT: Applies stratification on the target variable using a custom discretization of the target variable.
    I.e. uses the sklearn StratifiedKFold cross validation but for a continuous target variable instead of a multi-class target variable.
    - GROUP: Applies grouping information on the samples. I.e. uses the sklearn GroupKFold cross validation.
    - STRATGROUP: Uses the sklearn StratifiedGroupKFold cross validation.
    - CUSTOMSTRATGROUP: Applies stratification to both the target variable and the grouping information.
    I.e. uses the sklearn StratifiedGroupKFold cross validation but for a continuous target variable instead of a multi-class target variable.

    Args:

    Returns:

    """

    KFOLD = "KFold"
    GROUP = "GroupKFold"
    CUSTOMSTRAT = "CustomStratifiedKFold"
    STRATGROUP = "StratifiedGroupKFold"
    CUSTOMSTRATGROUP = "CustomStratifiedGroupKFold"


class CustomStratifiedGroupKFold(BaseCrossValidator):
    """sklearn's StratifiedGroupKFold adapted for continuous target variables."""

    def __init__(self, n_splits, shuffle=True, random_state=42, groups=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.groups = groups

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        The data is first grouped by groups and then split into n_splits folds. The folds are made by preserving the percentage of samples for each class.
        This is a variation of StratifiedGroupKFold that uses a custom discretization of the target variable.

        Args:
          X: Features
          y: target
          groups: Grouping/clustering variable (Default value = None)

        Returns:
            Iterator[tuple[ndarray, ndarray]]: Iterator over the indices of the training and test set.
        """
        self.sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        assert y is not None, "y cannot be None"
        kbins = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        if isinstance(y, pd.Series):
            y_cat = (
                kbins.fit_transform(y.to_numpy().reshape(-1, 1)).flatten().astype(int)
            )
            y_cat = pd.Series(y_cat, index=y.index)
        else:
            y_cat = kbins.fit_transform(y.reshape(-1, 1)).flatten().astype(int)  # type: ignore
        return self.sgkf.split(X, y_cat, groups)

    def get_n_splits(self, X, y=None, groups=None):
        """

        Args:
          X: 
          y:  (Default value = None)
          groups:  (Default value = None)

        Returns:
          int
        """
        return self.n_splits


class CustomStratifiedKFold(BaseCrossValidator):
    """sklearn's StratifiedKFold adapted for continuous target variables."""

    def __init__(self, n_splits, shuffle=True, random_state=42, groups=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.groups = groups

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        The data is first grouped by groups and then split into n_splits folds. The folds are made by preserving the percentage of samples for each class.
        This is a variation of StratifiedGroupKFold that uses a custom discretization of the target variable.

        Args:
          X: Features
          y: target
          groups: Grouping variable (Default value = None)

        Returns:
            Iterator[tuple[ndarray, ndarray]]: Iterator over the indices of the training and test set.
        """
        self.skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        assert y is not None, "y cannot be None"
        kbins = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        if isinstance(y, pd.Series):
            y_cat = (
                kbins.fit_transform(y.to_numpy().reshape(-1, 1)).flatten().astype(int)
            )
            y_cat = pd.Series(y_cat, index=y.index)
        else:
            y_cat = kbins.fit_transform(y.reshape(-1, 1)).flatten().astype(int)  # type: ignore
        # concatenate y_cat and groups such that the stratification is done on both
        y_concat = y_cat.astype(str) + "_" + groups.astype(str)
        return self.skf.split(X, y_concat)

    def get_n_splits(self, X, y=None, groups=None):
        """

        Args:
          X: 
          y:  (Default value = None)
          groups:  (Default value = None)

        Returns:
         int
        """
        return self.n_splits


def make_cross_val_split(
    *,
    groups: pd.Series | None,
    method: CrossValMethod,
    n_splits: int = 5,
    random_state: int = 42,
) -> Callable[..., Iterator[tuple[ndarray, ndarray]]]:
    """This function creates and returns a callable cross validation splitter based on the specified method.

    Args:
      groups: pd.Series | None: A pd.Series containing the grouping information for the samples.
      method: CrossValMethod: A CrossValMethod enum value specifying the cross validation method to use.
      n_splits: int: Number of splits (Default value = 5)
      random_state: int: A random seed to control random processes (Default value = 42)

    Returns:
      Callable: A callable cross validation splitter based on the specified method.

    Raises:
      TypeError: If the given method is not one of KFOLD

    """

    match method:
        case CrossValMethod.KFOLD:
            cross_val_obj = KFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return cross_val_obj.split

        case CrossValMethod.GROUP:
            if groups is None:
                raise ValueError("Groups must be specified for GroupKFold.")
            cross_val_obj = GroupKFold(n_splits=n_splits)
            return partial(cross_val_obj.split, groups=groups)

        case CrossValMethod.STRATGROUP:
            if groups is None:
                raise ValueError("Groups must be specified for StratGroupKFold.")
            cross_val_obj = StratifiedGroupKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return partial(cross_val_obj.split, groups=groups)

        case CrossValMethod.CUSTOMSTRATGROUP:
            if groups is None:
                raise ValueError("Groups must be specified for CustomStratGroupKFold.")
            cross_val_obj = CustomStratifiedGroupKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return partial(cross_val_obj.split, groups=groups)

        case CrossValMethod.CUSTOMSTRAT:
            if groups is None:
                raise ValueError("Groups must be specified for our StratifiedKFold.")
            cross_val_obj = CustomStratifiedKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return partial(cross_val_obj.split, groups=groups)
        case _:
            raise TypeError("Invalid Cross Validation method given.")
