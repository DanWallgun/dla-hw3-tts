#!/bin/bash

# install libraries
pip install -r requirements.txt

./scripts/download_git.sh
./scripts/download_ljspeech.sh
