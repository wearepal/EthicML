"""Run mypy and report how many errors there are."""


import sys

from mypy import api as mypy

MAX_ALLOWED_ERRORS = 25

# Also check types on the tests
RESULTS = mypy.run(["./tests/"])
print(RESULTS[0], end="")
print(RESULTS[1], end="")
NUM_ERRORS = len([line for line in RESULTS[0].split("\n") if ": error: " in line])
print(f"There are {NUM_ERRORS} errors. {MAX_ALLOWED_ERRORS} are allowed.")

if NUM_ERRORS <= MAX_ALLOWED_ERRORS:
    sys.exit(0)
else:
    sys.exit(1)
