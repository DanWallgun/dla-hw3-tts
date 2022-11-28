#!/bin/bash
python synthesize.py \
    --checkpoint-path="./model_new/current_model-state_dict-N.pth" \
    --text \
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest" \
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education" \
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space" \
    --duration-alpha 0.8 1.0 1.2 \
    --energy-alpha 0.8 1.0 1.2 \
    --pitch-alpha 0.8 1.0 1.2