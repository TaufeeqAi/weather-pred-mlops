from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements=f.read().splitlines()


setup(
    name="weather_prediction",
    version="0.01",
    author="Taufeeq",
    packages=find_packages(),
    install_requires= requirements,
)