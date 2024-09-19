import argparse
from pathlib import Path
import sys
import functools

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np

from data._utils import wav2feature
from data._utils_midi import midi2note, note2label
from utils import config


DIR_DATASET = ROOT / config.path.dataset
DIR_SYNCED = DIR_DATASET / "synced/"
DIR_ARRAY = DIR_DATASET / "array/"
DIR_ARRAY.mkdir(exist_ok=True)
DIR_NAME_PIANO = "piano/"

print = functools.partial(print, flush=True)


def main(args):
    songs = list(DIR_SYNCED.glob("*/"))
    songs = sorted(songs)
    n_songs = len(songs)

    for n, song in enumerate(songs, 1):
        print(f"{n}/{n_songs}: {song.name}", end=" ")
        create_label(song, args.overwrite)


def create_label(song, overwrite):
    dir_song = DIR_ARRAY / song.name
    if (not overwrite) and dir_song.exists():
        print("Already exists, skip.")
        return

    dir_song.mkdir(exist_ok=True)
    dir_song_piano = dir_song / DIR_NAME_PIANO
    dir_song_piano.mkdir(exist_ok=True)

    orig, = list(song.glob("*.wav"))
    pianos = sorted(list((song / DIR_NAME_PIANO).glob("*.mid")))

    spec = wav2feature(str(orig))
    np.save(dir_song / orig.stem, spec)
    labels = {}
    for piano in pianos:
        prefix = dir_song_piano / piano.stem
        if (not args.overwrite) and prefix.with_suffix(".npz").exists():
            continue
        label = get_label(piano)
        labels[prefix] = label

    for prefix, label in labels.items():
        np.savez(
            prefix,
            onset=label["onset"],
            offset=label["offset"],
            frames=label["frames"],
            velocity=label["velocity"],
        )
        print(".", end="")
    print(f" Done.")


def get_label(path_midi: Path):
    notes = midi2note(str(path_midi))
    label = note2label(notes)
    label = {
        "onset": np.array(label["onset"], dtype=np.float32),
        "offset": np.array(label["offset"], dtype=np.float32),
        "frames": (np.array(label["mpe"]) > 0.5).astype(np.uint8),
        "velocity": np.array(label["velocity"], dtype=np.uint8),
    }
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create labels for the dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    main(args)