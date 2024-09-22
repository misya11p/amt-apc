import argparse
import os

import torch
import torch.multiprocessing as mp

from _trainer import Trainer
from data import PianoCoversDataset


DEVICE_CUDA = torch.device("cuda")


def main(args):
    print("Start training.")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dataset = PianoCoversDataset(split=args.split)
    print(f"Dataset split: {args.split}.", flush=True)
    print(f"Number of samples: {len(dataset)}.", flush=True)

    trainer = Trainer(
        path_model=args.path_model,
        dataset=dataset,
        n_gpus=args.n_gpus,
        with_sv=not args.no_sv,
        no_load=args.no_load,
        freq_save=args.freq_save,
    )
    if args.n_gpus >= 2:
        print(f"Number of GPUs: {args.n_gpus}, using DDP.", flush=True)
        mp.spawn(
            trainer,
            nprocs=args.n_gpus,
            join=True,
        )
    else:
        print("Number of GPUs: 1", flush=True)
        trainer(DEVICE_CUDA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the model.")
    parser.add_argument("--path_model", type=str, default=None, help="Path to the base model. Defaults to CONFIG.PATH.AMT.")
    parser.add_argument("--n_gpus", type=int, default=2, help="Number of GPUs to use. Defaults to 2.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use: 'train' or 'test' or 'all'. Defaults to 'train'.")
    parser.add_argument("--no_sv", action="store_true", help="Do not use the style vector.")
    parser.add_argument("--no_load", action="store_true", help="Do not load the base model.")
    parser.add_argument("--freq_save", type=int, default=100, help="Frequency to save the model and logs. Defaults to 100.")
    args = parser.parse_args()
    main(args)
