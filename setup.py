from setuptools import setup, find_packages

setup(
    name='EthicML',
    version='0.1.0',
    author='O. Thomas',
    author_email="olliethomas86@gmail.com",
    description="A toolkit for understanding and researching algorithmic bias",
    url="https://github.com/predictive-analytics-lab/EthicML",
    packages=find_packages(),
    package_data={'ethicml.data': ['csvs/*.csv']},
    python_requires=">=3.6",
    install_requires=[
        "imageio",
        "matplotlib >= 3.0.2",
        "numpy >= 1.14.2",
        "pandas >= 0.24.0",
        "scikit_learn >= 0.20.1",
        "seaborn",
        "torch >= 0.4.1",
        "pyarrow >= 0.11",
        "numba",
        "fairlearn >= 0.2.0",
        "GitPython",
        "tqdm",
        "pipenv"
    ],
    extras_require={
        'dev': [  # use `pip install ethicml[dev]` to install development packages
            "pylint >= 2.0",
            "pytest >= 3.3.2",
            "pytest-cov >= 2.6.0",
            "mypy",
        ],
    },
)
