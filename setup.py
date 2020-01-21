from setuptools import setup, find_packages

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = "0"
_MINOR_VERSION = "1"
_PATCH_VERSION = "0"

_VERSION_SUFFIX = "dev"

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
    packages=find_packages(),
    package_data={"ethicml.data": ["csvs/*.csv"], "ethicml": ["py.typed"]},
    python_requires=">=3.6",
    install_requires=[
        'dataclasses;python_version<"3.7"',  # dataclasses are in the stdlib in python>=3.7
        "fairlearn >= 0.4.0",
        "GitPython >= 2.1.11",
        "matplotlib >= 3.0.2",
        "numpy >= 1.14.2",
        "pandas >= 0.24.0",
        "pipenv >= 2018.11.26",
        "pillow < 7.0",
        "pyarrow >= 0.11",
        "scikit_learn >= 0.20.1",
        "seaborn >= 0.9.0",
        "tqdm >= 4.31.1",
        "typed-argument-parser == 1.4",
        "typing-extensions >= 3.7.2",
    ],
    extras_require={
        "ci": [
            "pytest >= 3.3.2",
            "pytest-cov >= 2.6.0",
            "torch >= 1.1.0, <= 1.1.0.post2",
            "torchvision == 0.3.0",
        ],
        # use `pip install EthicML[dev]` to install development packages
        "dev": [
            "black",
            "data-science-types",
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
