#!/bin/bash

set -e #exit immediately after one of the commands failed

pip install --upgrade pip

pip install jupyter \
            matplotlib \
            numpy \
            pandas \
            pytest \
            scikit-learn \
            tqdm \
            transformers \