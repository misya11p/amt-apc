import argparse
from pathlib import Path
import json
import multiprocessing

import torch

# from data import wav2feature


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)
DIR_NAME_SYNCED = "audio-synced/"
DIR_NAME_SPEC = "spec/"
DIR_NAME_PIANO = "piano/"
PATH_STYLES = "dataset/styles.json"


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_SYNCED
    dir_output = dir_dataset / DIR_NAME_SPEC
    dir_output.mkdir(exist_ok=True)
    with open(PATH_STYLES, "w") as f:
        json.dump({"max_id": -1, "songs": {}}, f)

    save_args = []
    for song in dir_input.glob("*/"):
        save_args.append((song, dir_output, args.overwrite))

    with multiprocessing.Pool(args.n_processes) as pool:
        pool.starmap(_save_feature, save_args)


def _save_feature(dir_song: str, dir_output: str, overwrite: bool = False):
    """
    Convert the audio files in given song directory to features and save
    them in a single '.pth' file.

    Args:
        dir_song (str): Path to the song directory.
        dir_output (str): Path to the output directory.
    """
    file_output = (dir_output / dir_song.name).with_suffix(".pth")
    # if (not overwrite) and file_output.exists():
    #     print(f"'{file_output}' already exists, skipping.")
    #     return

    # features = []
    # orig = next(dir_song.glob("*.wav"))
    # feature_orig = wav2feature(str(orig))
    # features.append(feature_orig)

    covers = list((dir_song / DIR_NAME_PIANO).glob("*.wav"))
    # for cover in covers:
    #     feature = wav2feature(str(cover))
    #     features.append(feature)

    # min_len = len(min(features, key=len))
    # features = [feature[:min_len] for feature in features]
    # features = torch.stack(features)

    # torch.save(features, file_output)
    _assign_styleid(dir_song.name, len(covers))
    print(f"Saved '{file_output}'")


def _assign_styleid(title: str, n_covers: int):
    with open(PATH_STYLES, "r") as f:
        styles = json.load(f)
    max_id = styles["max_id"]

    styles["songs"][title] = list(range(max_id + 1, max_id + n_covers + 1))
    styles["max_id"] = max_id + n_covers

    with open(PATH_STYLES, "w") as f:
        json.dump(styles, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train dataset.")
    parser.add_argument("-d", "--path_dataset", type=str, default="dataset/", help="Path to the datasets directory. Defaults to '../datasets/'.")
    parser.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use. Defaults to 1.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    main(args)
