#/bin/bash

accelerate launch --num_processes 1 train_accelerate.py --config="configs/instructspeech.yaml"
