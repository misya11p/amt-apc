import argparse
from pathlib import Path
import json
import multiprocessing

import torch

from data import wav2feature


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)


DIR_NAME_SYNCED = "audio-synced/"
DIR_NAME_SPEC = "spec/"
DIR_NAME_PIANO = "piano/"


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_SYNCED
    dir_output = dir_dataset / DIR_NAME_SPEC
    dir_output.mkdir(exist_ok=True)

    save_args = [(song, dir_output) for song in dir_input.glob("*/")]
    with multiprocessing.Pool(args.n_processes) as pool:
        pool.starmap(save_feature, save_args)


def save_feature(dir_song, dir_output):
    features = []
    orig = next(dir_song.glob("*.wav"))
    feature_orig = wav2feature(str(orig))
    features.append(feature_orig)

    for piano in (dir_song / DIR_NAME_PIANO).glob("*.wav"):
        feature = wav2feature(str(piano))
        features.append(feature)

    min_len = len(min(features, key=len))
    features = [feature[:min_len] for feature in features]
    features = torch.stack(features)

    file_output = (dir_output / dir_song.name).with_suffix(".pth")
    torch.save(features, file_output)
    print(f"Saved '{file_output}' .")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train dataset.")
    parser.add_argument("-d", "--path_dataset", type=str, default="dataset/", help="Path to the datasets directory. Defaults to '../datasets/'.")
    parser.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use. Defaults to 1.")
    args = parser.parse_args()
    main(args)