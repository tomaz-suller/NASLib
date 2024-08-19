import os
import sys
import subprocess
from importlib import import_module
from setuptools import setup, find_packages


def install_if_not_found(module: str, install_from: str, user: bool = False):
    try:
        import_module(module)
    except ModuleNotFoundError:
        print(f"Installing {module}")
        python_command = f"{sys.executable} -m pip install --upgrade".split()
        if user:
            python_command.append("--user")
        python_command.append(install_from)
        subprocess.run(python_command, check=False)


cwd = os.path.dirname(os.path.abspath(__file__))

version_path = os.path.join(cwd, "naslib", "__version__.py")
with open(version_path) as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

install_if_not_found(
    "nasbench301",
    "git+https://github.com/tomaz-suller/nasbench301.git@fix-naslib",
)

install_if_not_found(
    "ConfigSpace",
    "git+https://github.com/tomaz-suller/ConfigSpace.git@fix-naslib",
)

print("-- Building version " + version)
print(
    "-- Note: by default installs pytorch-cpu version (1.9.0), update to torch-gpu by following instructions from: https://pytorch.org/get-started/locally/"
)

setup(
    name="naslib",
    version=version,
    description="NASLib: A modular and extensible Neural Architecture Search (NAS) library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AutoML Freiburg",
    author_email="zelaa@cs.uni-freiburg.de",
    url="https://github.com/automl/NASLib",
    license="Apache License 2.0",
    classifiers=["Development Status :: 1 - Beta"],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    platforms=["Linux"],
    install_requires=requirements,
    keywords=["NAS", "automl"],
    test_suite="pytest",
)
