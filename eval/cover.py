import sys; sys.path.append("./")
import argparse
from pathlib import Path
import json
import random

import torch
from tqdm import tqdm

from models import Pipeline


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("models/config.json", "r") as f:
    CONFIG = json.load(f)
PATH_PC = CONFIG["default"]["pc"]
SV_DIM = CONFIG["model"]["sv_dim"]


def main(args):
    dir_input = Path(args.dir_original)
    dir_output = Path(args.dir_output)
    dir_output.mkdir(exist_ok=True)
    with open(args.path_params, "r") as f:
        params = json.load(f)

    device = torch.device(args.device) if args.device else DEFAULT_DEVICE
    pipeline = Pipeline(path_model=args.model, device=device, sv_dim=SV_DIM)

    with open(args.path_style_vector) as f:
        style_vectors = list(json.load(f)["style_vector"].values())

    songs = list(dir_input.glob("*.wav"))
    songs = sorted(songs)
    for song in tqdm(songs):
        path_midi = dir_output / f"{song.stem}.mid"
        sv = torch.tensor(random.choice(style_vectors)).to(device)
        pipeline.wav2midi(
            path_input=str(song),
            path_output=str(path_midi),
            sv=sv,
            thred_onset=params["threshold"]["onset"],
            thred_offset=params["threshold"]["offset"],
            thred_mpe=params["threshold"]["mpe"],
            min_length=params["min_duration"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dir_original", type=str, default="./eval/data/songs")
    parser.add_argument("-o", "--dir_output", type=str, default="./eval/data/cover_midi")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--path_params", type=str, default="./eval/params.json")
    parser.add_argument("--path_style_vector", type=str, default="./data/style_vector.json")
    args = parser.parse_args()
    main(args)