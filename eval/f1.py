import sys; sys.path.append(".")
import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PianoCoversDataset
from models import load_model
from train import loss_fn


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open("models/config.json", "r") as f:
    CONFIG = json.load(f)
SV_DIM = CONFIG["model"]["sv_dim"]


def main(args):
    device = torch.device(args.device) if args.device else DEFAULT_DEVICE
    model = load_model(device, path_model=args.path_model, sv_dim=SV_DIM)

    dir_dataset = Path(args.dir_dataset)
    dataset = PianoCoversDataset(dir_dataset, use="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    loss, f1 = get_f1(model, dataloader, device)

    print(f"loss: {loss}")
    print(f"f1: {f1}")
    with open(args.path_output, "w") as f:
        f.write(f"loss: {loss}\n")
        f.write(f"f1: {f1}\n")


@torch.no_grad()
def get_f1(model, dataloader, device):
    all_loss = 0
    all_f1 = 0

    model.eval()
    for batch in tqdm(dataloader):
        spec, sv, onset, offset, mpe, velocity = batch
        spec = spec.to(device)
        sv = sv.to(device)
        onset = onset.to(device)
        offset = offset.to(device)
        mpe = mpe.to(device)
        velocity = velocity.to(device)

        pred = model(spec, sv)
        label = onset, offset, mpe, velocity
        loss, f1 = loss_fn(pred, label)

        all_loss += loss.item()
        all_f1 += sum(f1) / 3

    loss = all_loss / len(dataloader)
    f1 = all_f1 / len(dataloader)
    return loss, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_dataset", type=str, default="./dataset/dataset/")
    parser.add_argument("-o", "--path_output", type=str, default="./eval/f1.txt")
    parser.add_argument("--path_model", type=str, default="apc.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)
