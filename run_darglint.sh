#!/usr/bin/env sh

NUM_ALLOWED_DARGLINT_ERRS=65

echo "Running..."
darglint --verbosity 2 --no-exit-code --docstring-style sphinx -z long ethicml > darglint.out
cat darglint.out
num_errors=$(wc -l < darglint.out)
echo "Number of errors: $num_errors"

# if the reported number is greater (gt) than the allowed number, return with error code
if [ "$num_errors" -gt "$NUM_ALLOWED_DARGLINT_ERRS" ]; then
    exit 1;
fi
