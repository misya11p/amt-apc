import argparse
from pathlib import Path
import time
import functools
from collections import OrderedDict

import torch

from models import Pipeline


print = functools.partial(print, flush=True)

DIR_NAME_SYNCED = "synced/"
DIR_NAME_LABEL = "label/"
DIR_NAME_PIANO = "piano/"
DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_SYNCED
    dir_output = dir_dataset / DIR_NAME_LABEL
    dir_output.mkdir(exist_ok=True)

    songs = list(dir_input.glob("*/"))
    songs = sorted(songs)
    n_songs = len(songs)

    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    amt = Pipeline(device=device, amt=True, path_model=args.path_model_amt)

    for n, song in enumerate(songs, 1):
        name_song = song.name
        print(f"{n}/{n_songs}: {name_song}", end=" ")
        time_start = time.time()
        dir_song = dir_output / name_song
        if (not args.overwrite) and dir_song.exists():
            print("Already exists, skip.")
            continue
        dir_song.mkdir(exist_ok=True)

        pianos = sorted(list((song / DIR_NAME_PIANO).glob("*.wav")))
        for piano in pianos:
            orig, = list(song.glob("*.wav"))
            onset, offset, mpe, velocity = transcript(piano, amt)
            path_save = (dir_song / piano.stem).with_suffix(".pth")
            state_dict = {
                "onset": onset,
                "offset": offset,
                "mpe": mpe,
                "velocity": velocity,
                "info": {
                    "path_original": str(orig),
                }
            }
            torch.save(OrderedDict(state_dict), path_save)
            print(".", end="")
        print(f" Done ({time.time()-time_start:.2f}s)")


def transcript(path_input, amt):
    feature = amt.wav2feature(str(path_input))
    _, _, _, _, onset, offset, mpe, velocity = amt.transcript(feature)
    onset = (torch.from_numpy(onset) > 0.5).byte()
    offset = (torch.from_numpy(offset) > 0.5).byte()
    mpe = (torch.from_numpy(mpe) > 0.5).byte()
    return onset, offset, mpe, velocity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--path_model_amt", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)