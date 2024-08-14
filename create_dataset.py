import argparse
from pathlib import Path
import json
import functools

import numpy as np


print = functools.partial(print, flush=True)

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
    piano2orig = {}

    songs = list(dir_input.glob("*/"))
    songs = sorted(songs)
    n_songs = len(songs)
    for ns, song in enumerate(songs, 1):
        name_song = song.name
        print(f"{ns}/{n_songs}: {name_song}", end=" ")
        dir_piano = song / DIR_NAME_PIANO

        orig, = list(song.glob("*.npy"))
        spec = np.load(orig)
        length_song = len(spec) - MARGIN
        idxs = range(0, length_song, N_FRAMES)
        n_dig = len(str(len(idxs)))
        for ns, i in enumerate(idxs):
            spec_block = (spec[i:i + N_FRAMES + MARGIN]).T
            sid = str(ns).zfill(n_dig)
            path = dir_spec / f"{orig.stem}_{sid}"
            if (not args.overwrite) and Path(path).exists():
                continue
            np.save(path, spec_block)

        pianos = list(dir_piano.glob("*.npz"))
        pianos = sorted(pianos)
        for piano in pianos:
            piano2orig[piano.stem] = orig.stem
            midi = np.load(piano)
            midi_stack = np.stack((
                midi["onset"],
                midi["offset"],
                midi["mpe"],
                midi["velocity"],
            ))

            kwargs = []
            for ns, i in enumerate(range(0, length_song, N_FRAMES)):
                midi_block = (midi_stack[:, i:i + N_FRAMES])
                sid = str(ns).zfill(n_dig)
                path = dir_label / f"{piano.stem}_{sid}"
                if (not args.overwrite) and path.exists():
                    continue

                kwargs.append({
                    "path": path,
                    "data": {
                        "onset": midi_block[0].T,
                        "offset": midi_block[1].T,
                        "mpe": midi_block[2].T,
                        "velocity": midi_block[3].T,
                    }
                })
            if ne := args.rm_ends:
                kwargs = kwargs[ne:-ne]
            for kw in kwargs:
                np.savez(kw["path"], **kw["data"])

            print(".", end="")
        print(f" Done.")

    with open(PATH_PIANO2ORIG, "w") as f:
        json.dump(piano2orig, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train dataset.")
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/", help="Path to the datasets directory. Defaults to './datasets/'.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--rm_ends", type=int, default=2, help="Remove n segments from the beginning and the end of the song. Defaults to 2.")
    args = parser.parse_args()
    main(args)
