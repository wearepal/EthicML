"""Run all checks."""
import subprocess

import black
import pytest
from mypy import api as mypy
from pylint import lint as pylint

# pytest
print("############### pytest #################")
pytest.main(["-vv", "--cov=ethicml", "--cov-fail-under=80", "tests/"])
print("")

# pydocstyle
print("############### pydocstyle #################")
subprocess.call(
    [
        "pydocstyle",
        "--convention=google",
        "--add-ignore=D105,D107",
        "--ignore-decorators=implements|overload",
        "--count",
        "-e",
        "ethicml",
    ]
)
print("")

# pylint
print("############### pylint #################")
PYLINT_RESULTS = pylint.Run(["./ethicml/"], exit=False)
print("")

# pylint
print("############### pylint tests #################")
PYLINT_RESULTS = pylint.Run(["./tests/"], exit=False)
print("")

# mypy
print("###############  mypy  #################")
MYPY_RESULTS = mypy.run(["./ethicml/", "--warn-redundant-casts", "--show-error-context"])
print(MYPY_RESULTS[0], end="")
print(MYPY_RESULTS[1], end="")
print("Exit code of mypy: {}".format(MYPY_RESULTS[2]))

# mypy
print("############  mypy tests  ##############")
MYPY_RESULTS = mypy.run(
    ["./tests/", "--warn-redundant-casts", "--show-error-context", "--show-error-codes", "--pretty"]
)
print(MYPY_RESULTS[0], end="")
print(MYPY_RESULTS[1], end="")
print("Exit code of mypy: {}".format(MYPY_RESULTS[2]))

# black
print("############  black ##############")
black.main(["-l", "100", "-t", "py36", "-S", "./ethicml", "--config", ".black-config.toml"])
black.main(["-l", "100", "-t", "py36", "-S", "./tests", "--config", ".black-config.toml"])
print("")
