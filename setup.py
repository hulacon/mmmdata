# Everything else is in pyproject.toml. Only the package list is here, because
# pyproject's `find` can't add the mapped core.default_config package. mmmview
# is a top-level module (src/python/mmmview.py), not a package, so it is named
# separately.
from setuptools import find_packages, setup

setup(packages=find_packages("src/python") + ["core.default_config"],
      py_modules=["mmmview"])
