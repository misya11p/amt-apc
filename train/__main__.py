import argparse
import os

import torch
import torch.multiprocessing as mp

from _trainer import Trainer
from data import PianoCoversDataset


DEVICE_CUDA = torch.device("cuda")


def main(args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dataset = PianoCoversDataset(split=args.split)
    trainer = Trainer(
        path_model=args.path_model,
        dataset=dataset,
        n_gpus=args.n_gpus,
        with_sv=not args.no_sv,
        no_load=args.no_load,
        freq_save=args.freq_save,
    )
    if args.n_gpus >= 2:
        mp.spawn(
            trainer,
            nprocs=args.n_gpus,
            join=True,
        )
    else:
        trainer(DEVICE_CUDA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model", type=str, default=None)
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--split", type=str, default="train") # "train" or "test" or "all"
    parser.add_argument("--no_sv", action="store_true")
    parser.add_argument("--no_load", action="store_true")
    parser.add_argument("--freq_save", type=int, default=100)
    args = parser.parse_args()
    main(args)
