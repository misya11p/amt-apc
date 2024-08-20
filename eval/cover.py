import sys; sys.path.append("./")
import argparse
from pathlib import Path
import json
import random

import torch

from models import Pipeline


DIR_NAME_RAW = "raw/"
DIR_NAME_RAW = "eval/tmpraw/"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("models/config.json", "r") as f:
    CONFIG = json.load(f)
PATH_PC = CONFIG["default"]["pc"]
SV_DIM = CONFIG["model"]["sv_dim"]
with open("data/style_vector.json") as f:
    STYLE_VECTORS = list(json.load(f)["style_vector"].values())


def main(args):
    dir_dataset = Path(args.dir_dataset)
    dir_input = dir_dataset / DIR_NAME_RAW
    dir_tmp = dir_dataset / args.dir_tmp
    dir_tmp.mkdir(parents=True, exist_ok=True)
    with open(args.path_params, "r") as f:
        params = json.load(f)

    device = torch.device(args.device) if args.device else DEFAULT_DEVICE
    pipeline = Pipeline(path_model="pc.pth", device=device, sv_dim=SV_DIM)

    songs = list(dir_input.glob("*/"))
    songs = sorted(songs)
    for song in songs:
        print(song.stem)
        orig, = list(song.glob("*.wav"))
        path_midi = dir_tmp / f"{orig.stem}.mid"
        sv = torch.tensor(random.choice(STYLE_VECTORS))
        pipeline.wav2midi(
            path_input=str(orig),
            path_output=str(path_midi),
            sv=sv,
            thred_onset=params["threshold"]["onset"],
            thred_offset=params["threshold"]["offset"],
            thred_mpe=params["threshold"]["mpe"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir_dataset", type=str, default="./dataset/", help="Directory of dataset")
    parser.add_argument("--dir_tmp", type=str, default="./eval/tmp/", help="Directory of temporary files")
    parser.add_argument("--model", type=str, default=None, help="Model file")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--path_params", type=str, default="./eval/params.json", help="Path of parameters")
    args = parser.parse_args()
    main(args)