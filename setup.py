from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Dan(i) Jones",
    author_email="dannes@bas.ac.uk",
    description="Unsupervised classification of Weddell Gyre profiles",
    url="https://github.com/so-wise/weddell_gyre_clusters",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
