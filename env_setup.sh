#!/bin/bash

set -e #exit immediately after one of the commands failed

pip install --upgrade pip

pip install numpy \
            pandas \
            matplotlib \
            jupyter \
            tqdm \