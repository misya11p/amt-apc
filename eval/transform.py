import argparse
from pathlib import Path

from midi2audio import FluidSynth


def main(args):
    dir_input = Path(args.dir_midi)
    dir_save = Path(args.dir_save)
    dir_save.mkdir(exist_ok=True)

    covers = list(dir_input.glob("*.mid"))
    covers = sorted(covers)

    if args.sound_font:
        fs = FluidSynth(sound_font=args.sound_font)
    else:
        fs = FluidSynth()

    for cover in covers:
        path_save = dir_save / f"{cover.stem}.wav"
        fs.midi_to_audio(str(cover), str(path_save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_midi", type=str, default="./eval/data/cover_midi/")
    parser.add_argument("--dir_save", type=str, default="./eval/data/cover_audio/")
    parser.add_argument("--sound_font", type=str, default=None)
    args = parser.parse_args()
    main(args)
