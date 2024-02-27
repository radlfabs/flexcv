import pathlib

import pytest
from mktestdocs import check_md_file
from neptune.exceptions import NeptuneInvalidApiTokenException

@pytest.mark.xfail(raises=NeptuneInvalidApiTokenException)
def test_readme_codeblocks_valid(fpath):
    check_md_file(fpath=pathlib.Path(".") / "README.md")
