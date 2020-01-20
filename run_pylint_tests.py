"""
To stop pylint breaking the build if <85% perfect
"""

import sys
from pylint.lint import Run

RESULTS = Run(["./tests/"], do_exit=False)
SCORE = RESULTS.linter.stats["global_note"]
print("Your code has been rated at {:.2f}/10".format(SCORE))

if SCORE >= 8.5:
    sys.exit(0)
else:
    sys.exit(1)
