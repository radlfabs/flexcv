import pytest
from flexcv.split import CrossValMethod, string_to_crossvalmethod


def test_string_to_crossvalmethod_valid():
    # Test string_to_crossvalmethod function with valid inputs
    assert string_to_crossvalmethod("KFold") == CrossValMethod.KFOLD
    assert string_to_crossvalmethod("GroupKFold") == CrossValMethod.GROUP
    assert string_to_crossvalmethod("StratifiedGroupKFold") == CrossValMethod.STRATGROUP
    assert (
        string_to_crossvalmethod("ContinuousStratifiedKFold")
        == CrossValMethod.CONTISTRAT
    )
    assert (
        string_to_crossvalmethod("ContinuousStratifiedGroupKFold")
        == CrossValMethod.CONTISTRATGROUP
    )
    assert (
        string_to_crossvalmethod("ConcatenatedStratifiedKFold")
        == CrossValMethod.CONCATSTRATKFOLD
    )


def test_string_to_crossvalmethod_invalid():
    # Test string_to_crossvalmethod function with invalid input
    with pytest.raises(TypeError):
        string_to_crossvalmethod("InvalidString")


def test_string_to_crossvalmethod_non_string():
    # Test string_to_crossvalmethod function with non-string input
    with pytest.raises(TypeError):
        string_to_crossvalmethod(123)
