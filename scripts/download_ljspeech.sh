#!/bin/bash

#download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1

gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/

# mel spectrograms
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz
echo $(ls mels | wc -l)

# energy
gdown https://drive.google.com/u/0/uc?id=1HIeHw78jIzZaF6X0v8qFIfbqUjDSIn3I
tar -xvf energy.tar.gz
echo $(ls energy | wc -l)

# energy
gdown https://drive.google.com/u/0/uc?id=1ynl4bu92Am-qprq0TbOcgp8jvkHD6acP
tar -xvf pitch.tar.gz
echo $(ls pitch | wc -l)

# download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null