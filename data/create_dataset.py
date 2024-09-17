import argparse
from pathlib import Path
import sys
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np

from data._utils import preprocess_feature
from utils import config, info


DIR_DATASET = ROOT / config.path.dataset
DIR_ARRAY = DIR_DATASET / "array/"
DIR_FINAL = DIR_DATASET / "dataset/"
DIR_SPEC = DIR_FINAL / "spec/"
DIR_LABEL = DIR_FINAL / "label/"
DIR_NAME_PIANO = "piano/"

N_FRAMES = config.data.input.num_frame
MARGIN = config.data.input.margin_b + config.data.input.margin_f


def main(args):
    DIR_SPEC.mkdir(exist_ok=True, parents=True)
    DIR_LABEL.mkdir(exist_ok=True)

    songs = list(DIR_ARRAY.glob("*/"))
    is_train = {song.name: True for song in songs}
    random.shuffle(songs)
    for song in songs[:int(len(songs) * args.test_size)]:
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
            filename = DIR_SPEC / f"{orig.stem}_{sid}"
            if (not args.overwrite) and Path(filename).with_suffix(".npy").exists():
                continue
            np.save(filename, spec_block)

        pianos = list(dir_piano.glob("*.npz"))
        pianos = sorted(pianos)
        for piano in pianos:
            if not info[piano.stem].include_dataset:
                continue

            label = np.load(piano)
            label = {
                "onset": label["onset"],
                "offset": label["offset"],
                "frames": label["frames"],
                "velocity": label["velocity"],
            }
            label = align_length(label, length_song)

            save_args = []
            for ns, i in enumerate(range(0, length_song, N_FRAMES)):
                # (n_frames, n_bins)
                onset_block = label["onset"][i:i + N_FRAMES] # float [0, 1]
                offset_block = label["offset"][i:i + N_FRAMES] # float [0, 1]
                frames_block = label["frames"][i:i + N_FRAMES] # int {0, 1}
                velocity_block = label["velocity"][i:i + N_FRAMES] # int [0, 127]

                sid = str(ns).zfill(n_dig)
                prefix = DIR_LABEL / f"{piano.stem}_{sid}"
                if (not args.overwrite) and prefix.with_suffix(".npz").exists():
                    continue

                save_args.append({
                    "prefix": prefix,
                    "data": {
                        "onset": onset_block,
                        "offset": offset_block,
                        "frames": frames_block,
                        "velocity": velocity_block,
                    }
                })
            if ne := args.rm_ends:
                save_args = save_args[ne:-ne]
            for arg in save_args:
                np.savez(arg["prefix"], **arg["data"])
            info.update(piano.stem, {
                "n_segments": len(save_args),
                "split": "train" if is_train[name_song] else "test",
            })

            print(".", end="", flush=True)
        print(f" Done.", flush=True)


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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size. Defaults to 0.2.")
    parser.add_argument("--rm_ends", type=int, default=2, help="Remove n segments from the beginning and the end of the song. Defaults to 2.")
    args = parser.parse_args()
    main(args)
