import subprocess

subprocess.call(
    [
        "pydocstyle",
        "--convention=google",
        "--add-ignore=D105,D107",
        "--ignore-decorators=implements|overload",
        "--count",
        "-e",
        "ethicml",
    ]
)
