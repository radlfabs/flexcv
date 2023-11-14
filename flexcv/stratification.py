"""This module implements two stratificsation methods that can be used in contexts of regression of hierarchical (i.e. where the target is continuous and the data is grouped). 
"""
import pandas as pd
import numpy as np

from numpy.core._exceptions import UFuncTypeError
from sklearn.model_selection._split import (
    BaseCrossValidator,
    GroupsConsumerMixin,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import KBinsDiscretizer


class ContinuousStratifiedKFold(BaseCrossValidator):
    """Continuous Stratified k-Folds cross validator, i.e. it works with *continuous* target variables instead of multiclass targets.

    This is a variation of StratifiedKFold that

        - makes a copy of the target variable and discretizes it.
        - applies stratified k-folds based on this discrete target to ensure equal percentile distribution across folds
        - does not further use or pass this discrete target.
        - does not apply grouping rules.
    """

    def __init__(self, n_splits, shuffle=True, random_state=42, groups=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.groups = groups

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        The folds are made by preserving the percentage of samples for each class.
        This is a variation of StratifiedGroupKFold that uses a custom discretization of the target variable.

        Args:
          X (array-like): Features
          y (array-like): target
          groups (array-like): Grouping variable (Default value = None)

        Returns:
            (Iterator[tuple[ndarray, ndarray]]): Iterator over the indices of the training and test set.
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

        return self.skf.split(X, y_cat)

    def get_n_splits(self, X=None, y=None, groups=None):
        """

        Args:
          X (array-like): Features
          y (array-like): target values. (Default value = None)
          groups (array-like): grouping values. (Default value = None)

        Returns:
         (int) : The number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class ContinuousStratifiedGroupKFold(GroupsConsumerMixin, BaseCrossValidator):
    """Continuous Stratified Group k-Folds cross validator.
    This is a variation of StratifiedKFold that
        - makes a temporal discretization of the target variable.
        - apply stratified group k-fold based on the passed groups and the discretized target.
        - does not further use this discretized target
        - tries to preserve the percentage of samples in each percentile per group given the constraint of non-overlapping groups
    """

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
          X (array-like): Features
          y (array-like): target
          groups (array-like): Grouping/clustering variable (Default value = None)

        Returns:
            (Iterator[tuple[ndarray, ndarray]]): Iterator over the indices of the training and test set.
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

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Returns:
          (int): The number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class ConcatenatedStratifiedKFold(GroupsConsumerMixin, BaseCrossValidator):
    """Group Concatenated Continuous Stratified k-Folds cross validator.
    This is a variation of StratifiedKFold that uses a concatenation of target and grouping variable.

        - The target is discretized.
        - Each discrete target label is casted to type(str) and concatenated with the grouping label
        - Stratification is applied to this new temporal concatenated target
        - This preserves the group's *and* the targets distribution in each fold to be roughly equal to the input distribution
        - The procedure allows overlapping groups which could be interpreted as data leakage in many cases.
        - Population (i.e. the input data set) distribution is leaking into the folds' distribution.
    """

    def __init__(self, n_splits, shuffle=True, random_state=42, groups=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.groups = groups

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Applies target discretization, row-wise concatenation with the group-label, and stratification on this temporal concatenated column.

        Args:
          X (array-like): Features
          y (array-like): target
          groups (array-like): Grouping variable (Default value = None)

        Returns:
            (Iterator[tuple[ndarray, ndarray]]): Iterator over the indices of the training and test set.
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
        # elementwise concatenation of three arrays
        try:
            y_concat = y_cat.astype(str) + "_" + groups.astype(str)
        except UFuncTypeError:
            # Why easy when you can do it the hard way?
            y_concat = np.core.defchararray.add(
                np.core.defchararray.add(y_cat.astype(str), "_"), groups.astype(str)
            )

        return self.skf.split(X, y_concat)

    def get_n_splits(self, X=None, y=None, groups=None):
        """

        Args:
          X (array-like): Features
          y (array-like): target values. (Default value = None)
          groups (array-like): grouping values. (Default value = None)

        Returns:
         (int) : The number of splitting iterations in the cross-validator.
        """
        return self.n_splits
