#!/usr/bin/env bash
python -m pip install isort autopep8 black
isort ./
autopep8 --in-place --recursive ./
black -l 100 ./

read
