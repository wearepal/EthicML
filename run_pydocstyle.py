from subprocess import run

run(
    [
        "pydocstyle",
        "--convention=google",
        "--add-ignore=D105,D107",
        "--ignore-decorators=implements|overload",
        "--match=(?!(test_|__init__)).*\\.py",
        "--count",
        "-e",
        "ethicml",
    ],
    check=True,
)
