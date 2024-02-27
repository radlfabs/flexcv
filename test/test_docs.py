import pathlib

import pytest
from mktestdocs import check_md_file
from neptune.exceptions import NeptuneMissingApiTokenException

md_paths = [x for x in pathlib.Path("docs").glob("**/*.md")]
md_files = [str(x) for x in md_paths]

@pytest.mark.parametrize("fpath", md_paths, ids=md_files)
@pytest.mark.xfail(raises=NeptuneMissingApiTokenException)
def test_docs_codeblocks_valid(fpath):
    check_md_file(fpath=fpath)
