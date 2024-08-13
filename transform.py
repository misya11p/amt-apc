import argparse
from pathlib import Path
import time
import functools
from collections import OrderedDict

import torch
import numpy as np

from models import Pipeline
from data import preprocess_feature


print = functools.partial(print, flush=True)

DIR_NAME_SYNCED = "synced/"
DIR_NAME_TENSOR = "tensor/"
DIR_NAME_PIANO = "piano/"
DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_SYNCED
    dir_output = dir_dataset / DIR_NAME_TENSOR
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
        dir_song_piano = dir_song / DIR_NAME_PIANO
        dir_song_piano.mkdir(exist_ok=True)

        orig, = list(song.glob("*.wav"))
        pianos = sorted(list((song / DIR_NAME_PIANO).glob("*.wav")))

        feature_orig = amt.wav2feature(str(orig))
        lengths = [len(feature_orig)]
        features_piano = {}
        for piano in pianos:
            feature_piano = amt.wav2feature(str(piano))
            features_piano[piano] = feature_piano
            lengths.append(len(feature_piano))
        min_length = min(lengths)

        feature_orig = preprocess_feature(feature_orig[:min_length])
        path_save = dir_song / orig.stem
        np.save(path_save, feature_orig.numpy())

        for path, feature in features_piano.items():
            feature = feature[:min_length]
            onset, offset, mpe, velocity = transcript(feature, amt)
            path_save = dir_song_piano / path.stem
            np.savez(
                path_save,
                onset=onset.numpy(),
                offset=offset.numpy(),
                mpe=mpe.numpy(),
                velocity=velocity.numpy(),
            )
            print(".", end="")

        print(f" Done ({time.time()-time_start:.2f}s)")


def transcript(feature, amt):
    _, _, _, _, onset, offset, mpe, velocity = amt.transcript(feature)
    onset = (torch.from_numpy(onset) > 0.5).byte()
    offset = (torch.from_numpy(offset) > 0.5).byte()
    mpe = (torch.from_numpy(mpe) > 0.5).byte()
    velocity = torch.from_numpy(velocity).byte()
    return onset, offset, mpe, velocity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--path_model_amt", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)