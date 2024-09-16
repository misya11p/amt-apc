import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import torch
from tqdm import tqdm

from models import Pipeline
from utils import config


DIR_DATASET = root / config.dataset.dir
DIR_SYNCED = DIR_DATASET / "synced/"
DIR_NAME_PIANO = "piano/"
DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    piano_wavs = list(DIR_SYNCED.glob("*/piano/*.wav"))
    piano_wavs = sorted(piano_wavs)
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    amt = Pipeline(device=device, amt=True, path_model=args.path_amt)

    for piano_wav in tqdm(piano_wavs):
        piano_midi = piano_wav.with_suffix(".mid")
        if (not args.overwrite) and piano_midi.exists():
            continue
        amt.wav2midi(str(piano_wav), str(piano_midi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transcribe piano audio to MIDI.")
    parser.add_argument("--device", type=str, default=None, help="Device to use. Defaults to auto (CUDA if available else CPU).")
    parser.add_argument("--path_amt", type=str, default=None, help="Path to the AMT model. Defaults to the model in config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MIDI files.")
    args = parser.parse_args()
    main(args)