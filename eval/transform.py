import argparse
from pathlib import Path

from midi2audio import FluidSynth


def main(args):
    dir_input = Path(args.dir_input)
    dir_output = Path(args.dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    covers = list(dir_input.glob("*.mid"))
    covers = sorted(covers)

    fs = FluidSynth()
    for cover in covers:
        path_output = dir_output / f"{cover.stem}.wav"
        fs.midi_to_audio(str(cover), str(path_output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dir_input", type=str, default="./eval/tmp")
    parser.add_argument("-o", "--dir_output", type=str, default="./eval/covers/")
    args = parser.parse_args()
    main(args)