import subprocess

subprocess.call(
    [
        "pydocstyle",
        "--convention=google",
        "--add-ignore=D105",
        "--ignore-decorators=implements|overload",
        "--count",
        "-e",
        "ethicml",
    ]
)
