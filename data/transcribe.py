import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
from tqdm import tqdm

from models import Pipeline
from utils import config


DIR_DATASET = ROOT / config.dataset.dir
DIR_SYNCED = DIR_DATASET / "synced/"
DIR_NAME_PIANO = "piano/"
DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    piano_wavs = list(DIR_SYNCED.glob("*/piano/*.wav"))
    piano_wavs = sorted(piano_wavs)
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    amt = Pipeline(path_model=args.path_amt, device=device, amt=True)

    for piano_wav in tqdm(piano_wavs):
        piano_midi = piano_wav.with_suffix(".mid")
        if (not args.overwrite) and piano_midi.exists():
            continue
        amt.wav2midi(str(piano_wav), str(piano_midi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transcribe piano audio to MIDI.")
    parser.add_argument("--device", type=str, default=None, help="Device to use. Defaults to auto (CUDA if available else CPU).")
    parser.add_argument("--path_amt", type=str, default=None, help="Path to the AMT model. Defaults to None (use the default model).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MIDI files.")
    args = parser.parse_args()
    main(args)
