from setuptools import setup, find_packages

setup(
    name="pyac",
    version="0.0.1",
    description="Finite Differences Time Domain in visco-elastic mediums",
    author="Quentin Brateau",
    author_email="quentin.brateau@ensta-bretagne.fr",
    packages=find_packages(include=["pyac", "pyac.*"]),
    install_requires=["cupy", "numba", "numpy", "matplotlib"]
)