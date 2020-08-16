#!/bin/bash

path=egs/debug/tr
if [[ ! -e $path ]]; then
    mkdir -p $path
fi
python3 -m denoiser.audio dataset/debug/noisy > $path/noisy.json
python3 -m denoiser.audio dataset/debug/clean > $path/clean.json
