"""
In order to switch cross validation split methods dynamically we need to implement a function that returns the correct cross validation split function.
This is necessary because the split methods may have different numbers of arguments.
This module also implements a custom stratified cross validation split method for continuous target variables and a custom stratified group cross validation split method for continuous target variables that incorporates grouping information.
"""

from enum import Enum
from functools import partial
from typing import Callable, Iterator

import pandas as pd
from numpy import ndarray
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupsConsumerMixin,
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from .stratification import (
    ContinuousStratifiedKFold,
    ContinuousStratifiedGroupKFold,
    ConcatenatedStratifiedKFold
)


class CrossValMethod(Enum):
    """Enum class to assign CrossValMethods to the cross_val() function.
    This is useful to return the correct splitting function depending on the cross val method.

    Members:
        - `KFOLD`: Regular sklearn `KFold` cross validation. No grouping information is used.
        - `GROUP`: Regular sklearn `GroupKFold` cross validation. Grouping information is used.
        - `STRAT`: Regular sklearn `StratifiedKFold` cross validation. No grouping information is used.
        - `STRATGROUP`: Regular sklearn `StratifiedGroupKFold` cross validation. Grouping information is used.
        - `CONTISTRAT`: Stratified cross validation for continuous targets. No grouping information is used.
        - `CONTISTRATGROUP`: Stratified cross validation for continuous targets. Grouping information is used.
        - `CONCATSTRATKFOLD`: Stratified cross validation. Leaky stratification on element-wise-concatenated target and group labels.
    """

    KFOLD = "KFold"
    GROUP = "GroupKFold"
    STRAT = "StratifiedKFold"
    STRATGROUP = "StratifiedGroupKFold"
    CONTISTRAT = "ContinuousStratifiedKFold"
    CONTISTRATGROUP = "ContinuousStratifiedGroupKFold"
    CONCATSTRATKFOLD = "ConcatenatedStratifiedKFold"


def string_to_crossvalmethod(method: str) -> CrossValMethod:
    """Converts a string to a CrossValMethod enum member.

    Args:
      method (str): The string to convert.

    Returns:
      (CrossValMethod): The CrossValMethod enum value.

    Raises:
      (ValueError): If the given string does not match any CrossValMethod.

    """
    keys = [e.value for e in CrossValMethod]
    values = [e for e in CrossValMethod]
    method_dict = dict(zip(keys, values))

    if method in method_dict:
        return method_dict[method]
    else:
        raise ValueError("Invalid Cross Validation method given.")


def make_cross_val_split(
    *,
    groups: pd.Series | None,
    method: CrossValMethod,
    n_splits: int = 5,
    random_state: int = 42,
) -> Callable[..., Iterator[tuple[ndarray, ndarray]]]:
    """This function creates and returns a callable cross validation splitter based on the specified method.

    Args:
      groups (pd.Series | None): A pd.Series containing the grouping information for the samples.
      method (CrossValMethod): A CrossValMethod enum value specifying the cross validation method to use.
      n_splits (int): Number of splits (Default value = 5)
      random_state (int): A random seed to control random processes (Default value = 42)

    Returns:
      (Callable): A callable cross validation splitter based on the specified method.

    Raises:
      (TypeError): If the given method is not one of KFOLD

    """
    
    match method:
        case CrossValMethod.KFOLD:
            kf = KFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return kf.split
        
        case CrossValMethod.STRAT:
            strat_skf = StratifiedKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return strat_skf.split
        
        case CrossValMethod.CONTISTRAT:
            conti_skf = ContinuousStratifiedKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return conti_skf.split
        
        case CrossValMethod.GROUP:
            gkf = GroupKFold(n_splits=n_splits)
            return partial(gkf.split, groups=groups)
        
        case CrossValMethod.STRATGROUP:
            strat_gkf = StratifiedGroupKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return partial(strat_gkf.split, groups=groups)

        case CrossValMethod.CONTISTRATGROUP:
            conti_sgkf = ContinuousStratifiedGroupKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return partial(conti_sgkf.split, groups=groups)

        case CrossValMethod.CONCATSTRATKFOLD:
            concat_skf = ConcatenatedStratifiedKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True
            )
            return partial(concat_skf.split, groups=groups)
        
        case _:

            is_cross_validator = isinstance(method, BaseCrossValidator)
            is_groups_consumer = isinstance(method, GroupsConsumerMixin)
            
            if is_cross_validator and is_groups_consumer:
                return partial(method.split, groups=groups)
            
            if is_cross_validator:
                return method.split
    
            if isinstance(method, Iterator):
                return method
            
            else:
                raise TypeError("Invalid Cross Validation method given.")


if __name__ == "__main__":
    pass