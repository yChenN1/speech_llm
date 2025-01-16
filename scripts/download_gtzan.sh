#!/bin/bash

wget -O genres.tar.gz https://huggingface.co/datasets/qiuqiangkong/gtzan/resolve/main/genres.tar.gz?download=true
mkdir -p ./datasets/gtzan
tar -zxvf genres.tar.gz -C ./datasets/gtzan/