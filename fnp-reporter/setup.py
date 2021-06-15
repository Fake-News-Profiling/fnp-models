from setuptools import setup

setup(
    name="FNP-Reporter",
    version="1.0.0",
    packages=["fnpreporter"],
    description="Reporter client for Fake News Profiling",
    long_description=open("README.md").read(),
    install_requires=[
        "jinja2",
    ],
)

