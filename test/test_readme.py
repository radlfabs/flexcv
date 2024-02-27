import pathlib

import pytest
from mktestdocs import check_md_file
from neptune.exceptions import NeptuneInvalidApiTokenException


# @pytest.mark.xfail(raises=OSError)
@pytest.mark.xfail(raises=NeptuneInvalidApiTokenException)
@pytest.mark.parametrize("fpath", pathlib.Path(".").glob("README.md"), ids=["README"])  # readme file
def test_readme_codeblocks_valid(fpath):
    # Test if the code blocks in the docs files are valid
    # ignore the following Exception NeptuneInvalidApiTokenException
    # as this is a valid exception
    check_md_file(fpath=fpath)
