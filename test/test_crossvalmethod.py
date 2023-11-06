from flexcv.split import CrossValMethod, string_to_crossvalmethod

def test_string_conversion():
    """This test checks that the string conversion of the cross validation methods is correct."""
    assert string_to_crossvalmethod("KFold") == CrossValMethod.KFOLD
    assert string_to_crossvalmethod("GroupKFold") == CrossValMethod.GROUP
    assert string_to_crossvalmethod("StratifiedGroupKFold") == CrossValMethod.STRATGROUP
    assert string_to_crossvalmethod("CustomStratifiedGroupKFold") == CrossValMethod.CUSTOMSTRATGROUP
