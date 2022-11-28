#!/bin/bash

# install libraries
pip install -r requirements.txt

./scripts/download_waveglow.sh
./scripts/download_git.sh
./scripts/download_trained.sh