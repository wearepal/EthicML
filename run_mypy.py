"""Run mypy and report how many errors there are"""

import sys
from mypy import api as mypy

MAX_ALLOWED_ERRORS = 5

RESULTS = mypy.run(['./ethicml/'])
print(RESULTS[0], end='')
print(RESULTS[1], end='')
NUM_ERRORS = len(RESULTS[0].split('\n')) - 1
print("There are {} errors. {} are allowed.".format(NUM_ERRORS, MAX_ALLOWED_ERRORS))

if NUM_ERRORS <= MAX_ALLOWED_ERRORS:
    sys.exit(0)
else:
    sys.exit(1)

# Also check types on the tests
RESULTS = mypy.run(['./tests/', '--check-untyped-defs'])
print(RESULTS[0], end='')
print(RESULTS[1], end='')
NUM_ERRORS = len(RESULTS[0].split('\n')) - 1
print("There are {} errors. {} are allowed.".format(NUM_ERRORS, MAX_ALLOWED_ERRORS))

if NUM_ERRORS <= MAX_ALLOWED_ERRORS:
    sys.exit(0)
else:
    sys.exit(1)
