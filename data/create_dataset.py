import argparse
from pathlib import Path
import json
import random

import numpy as np

from _utils import preprocess_feature


DIR_NAME_ARRAY = "array/"
DIR_NAME_DATA = "data/"
DIR_NAME_PIANO = "piano/"
DIR_NAME_SPEC = "spec/"
DIR_NAME_LABEL = "label/"
PATH_DB = "data/db.json"
with open(PATH_DB, "r") as f:
    db = json.load(f)

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
    is_train = {song.name: True for song in songs}
    random.shuffle(songs)
    for song in songs[:int(len(songs) * args.valid_size)]:
        is_train[song.name] = False

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
            info_piano = db[piano.stem]
            if info_piano["n_notes"] < args.min_notes:
                info_piano["in_dataset"] = False
                save(db)
                continue

            label = np.load(piano)
            label = {
                "onset": label["onset"],
                "offset": label["offset"],
                "mpe": label["mpe"],
                "velocity": label["velocity"],
            }
            label = align_length(label, length_song)

            save_args = []
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

                save_args.append({
                    "prefix": prefix,
                    "data": {
                        "onset": onset_block,
                        "offset": offset_block,
                        "mpe": mpe_block,
                        "velocity": velocity_block,
                    }
                })
            if ne := args.rm_ends:
                save_args = save_args[ne:-ne]
                included_range = [ne, len(idxs) - ne]
            else:
                included_range = [0, len(idxs)]
            for arg in save_args:
                np.savez(arg["prefix"], **arg["data"])
            info_piano["in_dataset"] = True
            info_piano["included_range"] = included_range
            info_piano["n_segments"] = len(save_args)
            info_piano["train"] = is_train[name_song]

            print(".", end="", flush=True)
        print(f" Done.", flush=True)


def save(db):
    with open(PATH_DB, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


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
    parser.add_argument("--valid_size", type=float, default=0.2, help="Validation size. Defaults to 0.2.")
    parser.add_argument("--min_notes", type=int, default=500, help="Minimum number of notes to keep the song. Defaults to 500.")
    parser.add_argument("--rm_ends", type=int, default=2, help="Remove n segments from the beginning and the end of the song. Defaults to 2.")
    args = parser.parse_args()
    main(args)
