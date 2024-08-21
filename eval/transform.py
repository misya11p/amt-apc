import argparse
from pathlib import Path

from midi2audio import FluidSynth


DIR_NAME_COVER_MIDI = "cover_midi"
DIR_NAME_COVER_AUDIO = "cover_audio"


def main(args):
    dir_data = Path(args.dir_data)
    dir_input = dir_data / DIR_NAME_COVER_MIDI
    dir_output = dir_data / DIR_NAME_COVER_AUDIO
    dir_output.mkdir(exist_ok=True)

    covers = list(dir_input.glob("*.mid"))
    covers = sorted(covers)

    fs = FluidSynth()
    for cover in covers:
        path_output = dir_output / f"{cover.stem}.wav"
        fs.midi_to_audio(str(cover), str(path_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir_data", type=str, default="./eval/data/")
    args = parser.parse_args()
    main(args)