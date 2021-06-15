# Fake-News Profiling Models
This package contains fake-news profiling models, as well as experimentation and evaluation tools.

## Build and Import
Build with `python setup.py sdist bdist_wheel`.

Import into a local virtual environment with `python setup.py install`.

## Packages
`evaluation` contains functions used for evaluating the fake-news classification models.

`experiments` contains `Experiment` classes used to test classification models as well as perform hyperparameter tuning
for models. The `ExperimentHandler` coordinates saving config data, loading datasets, executing experiments and 
reviewing results.

`models` contains baseline and final fake-news classification models.

`processing` contains tools for preprocessing text data as well as extracting readability, sentiment, or named-entity 
information from text data.
