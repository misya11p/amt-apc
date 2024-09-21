from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import torch
from midi2audio import FluidSynth
from tqdm import tqdm

from models import Pipeline
from data import SVSampler
from utils import info


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sv_sampler = SVSampler()


def main(args):
    dir_output = ROOT / args.dir_output
    dir_output.mkdir(exist_ok=True)
    device = torch.device(args.device) if args.device else DEFAULT_DEVICE

    # Create MIDI files from WAV files
    midis = cover(
        dir_output, args.path_model, device, args.with_sv, args.no_load
    )

    # Convert MIDI files to audio files
    midi2audio(midis, args.sound_font)


def cover(dir_output, path_model, device, with_sv, no_load, overwrite):
    pipeline = Pipeline(
        path_model=path_model,
        device=device,
        with_sv=with_sv,
        no_load=no_load,
    )

    songs = info.get_ids("test", orig=True)
    songs = sorted(songs)
    midis = []
    for song in tqdm(songs):
        path_input = info.id2path(song).raw
        if not path_input.exists():
            print(f"File not found: {path_input}")
            continue

        path_output = dir_output / f"{song}.mid"
        if path_output.exists() and not overwrite:
            midis.append(path_output)
            continue

        sv = sv_sampler.random()
        pipeline.wav2midi(
            path_input=str(path_input),
            path_output=str(path_output),
            sv=sv,
        )
        midis.append(path_output)
    return midis


def midi2audio(midis, sound_font):
    if sound_font:
        fs = FluidSynth(sound_font=sound_font)
    else:
        fs = FluidSynth()

    for midi in midis:
        path_save = midi.with_suffix(".wav")
        fs.midi_to_audio(str(midi), str(path_save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_output", "-o", type=str, default="eval/data/")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--path_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--with_sv", action="store_true")
    parser.add_argument("--no_load", action="store_true")
    parser.add_argument("--sound_font", type=str, default=None)
    args = parser.parse_args()
    main(args)