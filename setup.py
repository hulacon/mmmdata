# Everything else is in pyproject.toml. Only the package list is here, because
# pyproject's `find` can't add the mapped core.default_config package.
from setuptools import find_packages, setup

setup(packages=find_packages("src/python") + ["core.default_config"])
