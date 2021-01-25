"""Welcome to EthicML."""

from setuptools import find_packages, setup

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = "0"
_MINOR_VERSION = "2"
_PATCH_VERSION = "0"

_VERSION_SUFFIX = ""

# Example, '0.4.0-rc1'
version = ".".join([_MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION])
if _VERSION_SUFFIX:
    version = f"{version}-{_VERSION_SUFFIX}"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="EthicML",
    version=version,
    author="Predictive Analytics Lab - University of Sussex",
    author_email="olliethomas86@gmail.com",
    description="A toolkit for understanding and researching algorithmic bias",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/predictive-analytics-lab/EthicML",
    packages=find_packages(exclude=["tests.*", "tests"]),
    package_data={"ethicml.data": ["csvs/*.csv", "csvs/*.csv.zip"], "ethicml": ["py.typed"]},
    python_requires=">=3.6",
    install_requires=[
        'dataclasses;python_version<"3.7"',  # dataclasses are in the stdlib in python>=3.7
        "GitPython >= 2.1.11",
        "matplotlib >= 3.0.2, < 3.3.1",
        "numpy >= 1.14.2",
        "pandas >= 1.0",
        "pipenv >= 2018.11.26",
        "pillow",
        "scikit_learn >= 0.20.1",
        "seaborn >= 0.9.0",
        "teext",
        "tqdm >= 4.31.1",
        "typed-argument-parser == 1.4",
        "typing-extensions >= 3.7.2",
    ],
    extras_require={
        "all": ["fairlearn == 0.4.6"],
        "ci": [
            "fairlearn == 0.4.6",
            "pytest >= 3.3.2",
            "pytest-cov >= 2.6.0",
            "torch == 1.5.0",
            "torchvision == 0.6.0",
        ],
        # use `pip install EthicML[dev]` to install development packages
        "dev": [
            "black",
            "data-science-types",
            "isort == 5.5.4",
            "mypy",
            "pydocstyle",
            "pylint",
            "pytest",
            "pytest-cov",
            "pre-commit",
        ],
    },
    classifiers=[  # classifiers can be found here: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Typing :: Typed",
    ],
    zip_safe=False,
)
