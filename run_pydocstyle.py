import subprocess

subprocess.call(["pydocstyle", "--convention=google", "--count", "-e", "ethicml"])
