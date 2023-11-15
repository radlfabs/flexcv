import requests
import subprocess
import time
import pathlib
import pytest
from neptune.exceptions import NeptuneInvalidApiTokenException

from mktestdocs import check_md_file


@pytest.mark.xfail(raises=NeptuneInvalidApiTokenException)
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_docs_codeblocks_valid(fpath):
    # Test if the code blocks in the docs files are valid
    # ignore the following Exception NeptuneInvalidApiTokenException
    # as this is a valid exception

    check_md_file(fpath=fpath)


def test_docs_build_no_fails():
    # Test if the docs build without errors
    subprocess.check_call(["mkdocs", "build"])
