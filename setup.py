from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Dan Jones",
    author_email="author@example.com",
    description="Unsupervised classification of Weddell Gyre profiles",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
