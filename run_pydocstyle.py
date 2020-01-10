import subprocess

subprocess.call(
    [
        "pydocstyle",
        "--convention=google",
        "--ignore-decorators=implements",
        "--count",
        "-e",
        "ethicml",
    ]
)
