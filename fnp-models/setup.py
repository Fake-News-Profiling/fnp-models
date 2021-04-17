from setuptools import setup, find_packages

setup(
    name="fnp-models",
    version="1.0.0",
    packages=find_packages(),
    description="Models for Fake News Profiling",
    long_description=open("README.txt").read(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow == 2.4.1",
        "tensorflow_hub",
        "tf-models-official",
        "xgboost",
        "sklearn",
        "nltk",
        "stanza",
        "spacy",
        "textblob",
        "vaderSentiment",
        "keras-tuner",
        "pyphen",
        "demoji",
        "bs4",
        "unidecode",
    ],
)
