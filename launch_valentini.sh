#!/bin/bash

srun --gres gpu:4 --mem 500G --nodes 1 -n 1 --cpus-per-task 40 --partition dev --time 72:00:00 \
  python train.py \
  dset=valentini \
  demucs.causal=1 \
  demucs.hidden=48 \
  bandmask=0.2 \
  demucs.resample=4 \
  remix=1 \
  shift=8000 \
  shift_same=True \
  stft_loss=True \
  segment=4.5 \
  stride=0.5 \
  ddp=1

