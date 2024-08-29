import argparse
from pathlib import Path
import json

import numpy as np

from data import preprocess_feature


DIR_NAME_ARRAY = "array/"
DIR_NAME_DATA = "data/"
DIR_NAME_PIANO = "piano/"
DIR_NAME_SPEC = "spec/"
DIR_NAME_LABEL = "label/"
PATH_PIANO2ORIG = "data/piano2orig.json"

with open("models/config.json", "r") as f:
    CONFIG = json.load(f)["data"]
N_FRAMES = CONFIG["input"]["num_frame"]
MARGIN = CONFIG["input"]["margin_b"] + CONFIG["input"]["margin_f"]


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_ARRAY
    dir_output = dir_dataset / DIR_NAME_DATA
    dir_output.mkdir(exist_ok=True)
    dir_spec = dir_output / DIR_NAME_SPEC
    dir_spec.mkdir(exist_ok=True)
    dir_label = dir_output / DIR_NAME_LABEL
    dir_label.mkdir(exist_ok=True)

    songs = list(dir_input.glob("*/"))
    songs = sorted(songs)
    n_songs = len(songs)
    for ns, song in enumerate(songs, 1):
        name_song = song.name
        print(f"{ns}/{n_songs}: {name_song}", end=" ", flush=True)
        dir_piano = song / DIR_NAME_PIANO

        orig, = list(song.glob("*.npy"))
        spec = np.load(orig)
        spec = preprocess_feature(spec)
        length_song = len(spec) - MARGIN
        idxs = range(0, length_song, N_FRAMES)
        n_dig = len(str(len(idxs)))
        for ns, i in enumerate(idxs):
            spec_block = (spec[i:i + N_FRAMES + MARGIN]).T # (n_bins, n_frames)
            sid = str(ns).zfill(n_dig)
            filename = dir_spec / f"{orig.stem}_{sid}"
            if (not args.overwrite) and Path(filename).with_suffix(".npy").exists():
                continue
            np.save(filename, spec_block)

        pianos = list(dir_piano.glob("*.npz"))
        pianos = sorted(pianos)
        for piano in pianos:
            update_piano2orig(piano.stem, orig.stem)
            label = np.load(piano)
            label = align_length(label, length_song)

            args = []
            for ns, i in enumerate(range(0, length_song, N_FRAMES)):
                # (n_frames, n_bins)
                onset_block = label["onset"][i:i + N_FRAMES] # float [0, 1]
                offset_block = label["offset"][i:i + N_FRAMES] # float [0, 1]
                mpe_block = label["mpe"][i:i + N_FRAMES] # int {0, 1}
                velocity_block = label["velocity"][i:i + N_FRAMES] # int [0, 127]

                sid = str(ns).zfill(n_dig)
                prefix = dir_label / f"{piano.stem}_{sid}"
                if (not args.overwrite) and prefix.with_suffix(".npz").exists():
                    continue

                args.append({
                    "prefix": prefix,
                    "data": {
                        "onset": onset_block,
                        "offset": offset_block,
                        "mpe": mpe_block,
                        "velocity": velocity_block,
                    }
                })
            if ne := args.rm_ends:
                args = args[ne:-ne]
            for kw in args:
                np.savez(kw["prefix"], **kw["data"])

            print(".", end="", flush=True)
        print(f" Done.", flush=True)


def update_piano2orig(key, value):
    with open(PATH_PIANO2ORIG, "r") as f:
        piano2orig = json.load(f)
    piano2orig[key] = value
    with open(PATH_PIANO2ORIG, "w") as f:
        json.dump(piano2orig, f, indent=2)


def align_length(label, length):
    length_label = len(label["onset"])
    if length_label == length:
        pass
    elif length_label > length:
        for key in label.keys():
            label[key] = label[key][:length]
    else:
        for key in label.keys():
            label[key] = np.pad(label[key], ((0, length - length_label), (0, 0)))
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train dataset.")
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/", help="Path to the datasets directory. Defaults to './datasets/'.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--rm_ends", type=int, default=2, help="Remove n segments from the beginning and the end of the song. Defaults to 2.")
    args = parser.parse_args()
    main(args)
