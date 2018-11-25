from distutils.core import setup
from setuptools import find_packages

requires = ["six>=1.10.0"]

if __name__ == "__main__":
    setup(
        name="vusic-models",
        packages=find_packages(),
        author="DAWG",
        install_requires=requires,
        include_package_data=True,
    )
