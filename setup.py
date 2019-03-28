from setuptools import setup

setup(
    name='EthicML',
    version='0.1.0',
    author='O. Thomas',
    packages=['ethicml'],
    description='Framework for algorithmic fairness comparisons',
    python_requires=">=3.6",
    install_requires=[
        "imageio >= 2.4.1",
        "matplotlib >= 3.0.2",
        "numpy >= 1.14.2",
        "pandas >= 0.22.0",
        "pylint >= 1.8.2",
        "pytest >= 3.3.2",
        "pytest-cov >= 2.6.0",
        "scikit_learn >= 0.20.1",
        "seaborn >= 0.9.0",
        "torch >= 0.4.1",
        "typing >= 3.6.2",
        "fairlearn >= 0.2.0",
        "GitPython >= 2.1.11",
        "tqdm >= 4.31.1",
        "pipenv >= 2018.11.26"
    ],
)
