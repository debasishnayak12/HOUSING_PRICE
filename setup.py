from setuptools import find_packages,setup

setup(
    name="HOUSEPRICEPRED",
    version="0.0.1",
    author="debasish12",
    author_email="nayakdebasish7205@gmail.com",
    install_requires=['scikit-learn','pandas','numpy'],
    packages=find_packages()
)