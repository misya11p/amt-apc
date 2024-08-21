import sys; sys.path.append("./")
import argparse
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from _train import Trainer
from data import PianoCoversDataset


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dir_dataset = Path(args.dir_dataset)
    dataset = PianoCoversDataset(dir_dataset, use_all=args.use_all)
    trainer = Trainer(
        dataset=dataset,
        n_gpus=args.n_gpus,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        with_sv=not args.no_sv,
    )
    if args.n_gpus >= 2:
        mp.spawn(
            trainer,
            nprocs=args.n_gpus,
            join=True,
        )
    else:
        device = torch.device(args.device) if args.device else DEFAULT_DEVICE
        trainer(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir_dataset", type=str, default="./dataset/")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_sv", action="store_true")
    parser.add_argument("--use_all", action="store_true")
    args = parser.parse_args()
    main(args)