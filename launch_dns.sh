#!/bin/bash

srun --gres gpu:8 --mem 500G --nodes 1 -n 1 --cpus-per-task 40 --partition dev --time 72:00:00 \
  python train.py \
  dset=dns \
  demucs.causal=1 \
  demucs.hidden=64 \
  demucs.resample=4 \
  batch_size=128 \
  revecho=1 \
  segment=10 \
  stride=2 \
  shift=16000 \
  shift_same=True \
  epochs=250 \
  ddp=1
