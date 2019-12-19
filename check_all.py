"""Run all checks"""
import pytest
from pylint import lint as pylint
from mypy import api as mypy
import black

# pytest
print("############### pytest #################")
pytest.main(['-vv', '--cov=ethicml', '--cov-fail-under=80', 'tests/'])
print('')

# pylint
print("############### pylint #################")
PYLINT_RESULTS = pylint.Run(['./ethicml/'], do_exit=False)
print('')

# pylint
print("############### pylint tests #################")
PYLINT_RESULTS = pylint.Run(['./tests/'], do_exit=False)
print('')

# mypy
print("###############  mypy  #################")
MYPY_RESULTS = mypy.run(['./ethicml/', '--warn-redundant-casts', '--show-error-context'])
print(MYPY_RESULTS[0], end='')
print(MYPY_RESULTS[1], end='')
print("Exit code of mypy: {}".format(MYPY_RESULTS[2]))

# mypy
print("############  mypy tests  ##############")
MYPY_RESULTS = mypy.run(
    ['./tests/', '--warn-redundant-casts', '--show-error-context', '--show-error-codes', '--pretty']
)
print(MYPY_RESULTS[0], end='')
print(MYPY_RESULTS[1], end='')
print("Exit code of mypy: {}".format(MYPY_RESULTS[2]))

# black
print("############  black ##############")
black.main(['-l', '100', '-t', 'py36', '-S', './ethicml'])
black.main(['-l', '100', '-t', 'py36', '-S', './tests'])
print('')
