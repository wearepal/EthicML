Thanks for deciding to contribute to EthicML!

The 2 main ways to contribute to this project are raising an Issue, or opening a Pull Request.
  1. Raising an Issue: this is the main form of communication with the core developers. Come across something that you're not sure about, or want to discuss further? Raise an Issue. Is there a missing feature, or a bug, let us know via raising an Issue. Don't be afraid to open one; if it gets too much, we'll change these guidelines to say something else :D
  2. Opening a Pull Request: want to jump straight in? Sure, go ahead. If you're wanting to change something fundamental, such as the way our datastructures interact with each other, then it would be nice if you could raise an Issue first to describe in greater detail what you're thinking about. But we can always discuss this in a PR.
  

## Getting setup
1. Fork the repository, clone and `cd` to the newly cloned directory.
2. Setup and activate a new environment
3. Install the developer tools `pip install -e .[dev]`
4. (Optional) Install the pre-commit tools `pre-commit install`
5. Write your super cool new thing
6. (Optional) Check if your code would pass CI by running `python check_all.py` locally.

## CI
We use GitHub actions to test our code and make sure that we're keeping up the standards we set for ourselves.
Before your code can be accepted it must be formatted with Black, conform to PyLint, covered by tests, include MyPy type hints and be docuemnted.

This sounds like a lot, but it's super easy. If you neeed any help with this, we can add to your PR and get it in shape :D 
