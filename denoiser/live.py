# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import argparse
import sys

import sounddevice as sd
import torch

from .demucs import DemucsStreamer
from .pretrained import add_model_flags, get_model
from .utils import bold


def get_parser():
    parser = argparse.ArgumentParser(
        "denoiser.live",
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to 'Soundflower (2ch)' "
                    "(or the interface specified by --out)."
        )
    parser.add_argument(
        "-i", "--in", dest="in_",
        help="name or index of input interface.")
    parser.add_argument(
        "-o", "--out", default="Soundflower (2ch)",
        help="name or index of output interface.")
    add_model_flags(parser)
    parser.add_argument(
        "--sample_rate", type=int, default=16_000,
        help="Sample rate")
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "--device", default="cpu")
    parser.add_argument(
        "--dry", type=float, default=0.04,
        help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
             "but it might cause distortions. Default is 0.04")
    parser.add_argument("-t", "--num_threads", type=int)
    return parser


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = bold(f"Invalid {kind} audio interface {device}.\n")
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps


def main():
    args = get_parser().parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    model = get_model(args).to(args.device)
    model.eval()
    print("Model loaded.")
    streamer = DemucsStreamer(model, dry=args.dry)

    device_in = parse_audio_device(args.in_)
    caps = query_devices(device_in, "input")
    channels_in = min(caps['max_input_channels'], 2)
    stream_in = sd.InputStream(
        device=device_in,
        samplerate=args.sample_rate,
        channels=channels_in)

    device_out = parse_audio_device(args.out)
    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 2)
    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=args.sample_rate,
        channels=channels_out)

    first = True
    stream_in.start()
    stream_out.start()
    print("Ready to process audio.")
    cooldown = 0
    index = 0
    while True:
        try:
            index += 1
            if index % 100 == 0:
                streamer.reset_time_per_frame()
            length = streamer.total_length if first else streamer.demucs.total_stride
            first = False
            frame, overflow = stream_in.read(length)
            frame = torch.from_numpy(frame).mean(dim=1).to(args.device)
            with torch.no_grad():
                out = streamer.feed(frame[None])[0]
            if args.compressor:
                out = 0.99 * torch.tanh(out)
            out = out[:, None].repeat(1, channels_out)
            mx = out.abs().max().item()
            if mx > 1:
                print("Clipping!!")
            out.clamp_(-1, 1)
            out = out.cpu().numpy()
            underflow = stream_out.write(out)
            if overflow or underflow:
                if cooldown == 0:
                    cooldown = 10
                    tpf = 1000 * streamer.time_per_frame
                    print(f"Not processing audio fast enough, time per frame is {tpf:.1f}ms !")
                else:
                    cooldown -= 1
        except KeyboardInterrupt:
            print("Stopping")
            break
    stream_out.stop()
    stream_in.stop()


if __name__ == "__main__":
    main()
