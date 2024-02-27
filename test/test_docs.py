import pathlib

import pytest
from mktestdocs import check_md_file
from neptune.exceptions import NeptuneInvalidApiTokenException, NeptuneMissingApiTokenException


# @pytest.mark.xfail(raises=OSError)
@pytest.mark.xfail(raises=NeptuneMissingApiTokenException)
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_docs_codeblocks_valid(fpath):
    # Test if the code blocks in the docs files are valid
    # ignore the following Exception NeptuneInvalidApiTokenException
    # as this is a valid exception
    check_md_file(fpath=fpath)
