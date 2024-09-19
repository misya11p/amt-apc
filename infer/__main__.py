from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch

from models import Pipeline
from data import SVSampler
from utils import config


DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SV_SAMPLER = SVSampler()


def main(args):
    path_model = args.path_model or config.path.apc
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    pipeline = Pipeline(path_model, device)

    src = args.input
    if src.startswith("https://"):
        pass # todo

    sv = SV_SAMPLER.sample(params=args.style)
    pipeline.wav2midi(src, args.output, sv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input wav file or URL of YouTube video")
    parser.add_argument("-o", "--output", type=str, default="output.mid", help="Path to the output midi file. Defaults to 'output.mid'")
    parser.add_argument("-s", "--style", type=str, default="level2", help="Cover style. Valid values are 'level1', 'level2', and 'level3'. Defaults to 'level2'")
    parser.add_argument("--path_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)
