import argparse
from pathlib import Path
import time
import functools

import torch
import numpy as np

from data import wav2feature, preprocess_feature
from ._utils_midi import midi2note, note2label


print = functools.partial(print, flush=True)

DIR_NAME_SYNCED = "synced/"
DIR_NAME_ARRAY = "array/"
DIR_NAME_PIANO = "piano/"
DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_SYNCED
    dir_output = dir_dataset / DIR_NAME_ARRAY
    dir_output.mkdir(exist_ok=True)

    songs = list(dir_input.glob("*/"))
    songs = sorted(songs)
    n_songs = len(songs)

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
        pianos = sorted(list((song / DIR_NAME_PIANO).glob("*.mid")))

        spec = wav2feature(str(orig))
        # spec = preprocess_feature(spec)
        # length = len(spec)
        np.save(dir_song / orig.stem, spec)
        labels = {}
        for piano in pianos:
            prefix = dir_song_piano / piano.stem
            if (not args.overwrite) and prefix.with_suffix(".npz").exists():
                continue
            label = get_label(piano)
            labels[prefix] = label

        for prefix, label in labels.items():
            # labels = align_length(label, length)
            np.savez(
                prefix,
                onset=labels["onset"],
                offset=labels["offset"],
                mpe=labels["mpe"],
                velocity=labels["velocity"],
            )
            print(".", end="")
        print(f" Done ({time.time()-time_start:.2f}s)")


def get_label(path_midi):
    notes = midi2note(str(path_midi))
    label = note2label(notes)
    label = {
        "onset": np.array(label["onset"], dtype=np.float32),
        "offset": np.array(label["offset"], dtype=np.float32),
        "mpe": (np.array(label["mpe"]) > 0.5).astype(np.uint8),
        "velocity": np.array(label["velocity"], dtype=np.uint8),
    }
    return label


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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)