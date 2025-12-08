from setuptools import find_packages, setup

setup(
    name="memorax",
    version="0.1.0",  # follow semantic versioning
    author="Noah Farr",
    packages=find_packages(exclude=["popjaxrl"]),
)
