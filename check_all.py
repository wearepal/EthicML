"""Run all checks"""
import pytest
from pylint import lint as pylint
from mypy import api as mypy

# pytest
print("############### pytest #################")
pytest.main(['tests/'])
print('')

# pylint
print("############### pylint #################")
PYLINT_RESULTS = pylint.Run(['./ethicml/'], do_exit=False)
print('')

# mypy
print("###############  mypy  #################")
MYPY_RESULTS = mypy.run(['./ethicml/', '--warn-redundant-casts', '--show-error-context'])
print(MYPY_RESULTS[0], end='')
print(MYPY_RESULTS[1], end='')
