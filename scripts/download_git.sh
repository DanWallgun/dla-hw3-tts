#!/bin/bash

# we will use waveglow code, data and audio preprocessing from this repo
git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/text .
mv FastSpeech/waveglow/* waveglow/
mv FastSpeech/utils.py .
mv FastSpeech/glow.py .