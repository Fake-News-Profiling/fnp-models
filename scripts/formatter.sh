#!/usr/bin/env bash
python -m pip install isort autopep8 black
isort ./app
autopep8 --in-place --recursive ./app
black -l 100 ./app

read
