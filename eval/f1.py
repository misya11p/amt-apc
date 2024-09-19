from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PianoCoversDataset
from models import load_model
from train import loss_fn
from utils import config


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    path_model = args.path_model or config.path.apc
    device = torch.device(args.device) if args.device else DEFAULT_DEVICE
    model = load_model(
        device,
        path_model=path_model,
        with_sv=not args.no_sv,
        no_load=args.no_load,
    )

    dataset = PianoCoversDataset(split="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    loss, f1 = get_f1(model, dataloader, device)
    print(f"loss: {loss}")
    print(f"f1: {f1}")


@torch.no_grad()
def get_f1(model, dataloader, device):
    all_loss = 0
    all_f1 = 0

    model.eval()
    for batch in tqdm(dataloader):
        spec, sv, onset, offset, frame, velocity = batch
        spec = spec.to(device)
        sv = sv.to(device)
        onset = onset.to(device)
        offset = offset.to(device)
        frame = frame.to(device)
        velocity = velocity.to(device)

        pred = model(spec, sv)
        label = onset, offset, frame, velocity
        loss, f1 = loss_fn(pred, label)

        all_loss += loss.item()
        all_f1 += sum(f1) / 3

    loss = all_loss / len(dataloader)
    f1 = all_f1 / len(dataloader)
    return loss, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--no_sv", action="store_true")
    parser.add_argument("--no_load", action="store_true")
    args = parser.parse_args()
    main(args)
