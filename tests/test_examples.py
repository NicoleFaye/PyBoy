#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "gamewrapper, rom", [
        ("gamewrapper_tetris.py", pytest.lazy_fixture("tetris_rom")),
        ("gamewrapper_mario.py", pytest.lazy_fixture("supermarioland_rom")),
        ("gamewrapper_kirby.py", pytest.lazy_fixture("kirby_rom")),
    ]
)
def test_record_replay(gamewrapper, rom):
    script_path = os.path.dirname(os.path.realpath(__file__))
    base = Path(f"{script_path}/../extras/examples/")

    assert rom is not None
    p = subprocess.Popen([sys.executable, str(base / gamewrapper), str(Path(rom)), "--quiet"])
    return_code = p.wait()
    if return_code != 0:
        print(p.communicate())
        sys.exit(return_code)
