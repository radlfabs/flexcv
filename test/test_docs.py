import requests
import subprocess
import time
import pathlib
import pytest
import pkgutil
import inspect
import importlib
from neptune.exceptions import NeptuneInvalidApiTokenException
from mktestdocs import check_md_file, get_codeblock_members, check_docstring
import flexcv

@pytest.mark.xfail(raises=NeptuneInvalidApiTokenException)
@pytest.mark.parametrize('fpath', pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_docs_codeblocks_valid(fpath):
    # Test if the code blocks in the docs files are valid
    # ignore the following Exception NeptuneInvalidApiTokenException
    # as this is a valid exception
    
    check_md_file(fpath=fpath)
