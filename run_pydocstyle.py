import subprocess

subprocess.call(
    [
        "pydocstyle",
        "--convention=google",
        "--ignore-decorators=implements|overload",
        "--count",
        "-e",
        "ethicml",
    ]
)
