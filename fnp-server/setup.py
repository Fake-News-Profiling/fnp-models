from setuptools import setup, find_packages

setup(
    name="fnp-server",
    version="1.0.0",
    packages=find_packages(),
    description="Fake News Profiling Server",
    long_description=open("README.md").read(),
    install_requires=[
        "wheel",
        "flask",
        "dacite",
        "requests",
        "twython",
        # "fnp-models",
    ],
)
