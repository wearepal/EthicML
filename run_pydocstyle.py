from subprocess import run

run(
    [
        "pydocstyle",
        "--convention=pep257",
        "--add-ignore=D105,D107",
        "--ignore-decorators=implements|overload",
        "--match=(?!(test_|__init__)).*\\.py",
        "--count",
        "-e",
        "ethicml",
    ],
    check=True,
)
