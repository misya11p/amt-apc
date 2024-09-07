import sys; sys.path.append("./")
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from models import Pipeline


DIR_NAME_SYNCED = "synced/"
DIR_NAME_PIANO = "piano/"
DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_synced = dir_dataset / DIR_NAME_SYNCED
    piano_wavs = list(dir_synced.glob("*/piano/*.wav"))
    piano_wavs = sorted(piano_wavs)
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    amt = Pipeline(device=device, amt=True, path_model=args.path_amt)

    for piano_wav in tqdm(piano_wavs):
        piano_midi = piano_wav.with_suffix(".mid")
        if (not args.overwrite) and piano_midi.exists():
            continue
        amt.wav2midi(str(piano_wav), str(piano_midi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--path_amt", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)