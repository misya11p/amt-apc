from pathlib import Path
import sys
import argparse
import json
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from yt_dlp import YoutubeDL

from utils import config


DIR_RAW = ROOT / config.path.dataset / "raw"
FILE_SRC = ROOT / config.path.src
DIR_NAME_PIANO = "piano"


def main(args):
    path_src = args.path_src or FILE_SRC
    with open(path_src, "r") as f:
        src = json.load(f)

    for n, (title, movies) in enumerate(src.items(), 1):
        print(f"{n}/{len(src)} {title}")
        download(DIR_RAW / title, movies["original"], movies["pianos"])


def download(dir_song: Path, original: str, pianos: List[str]) -> None:
    """
    Download the audio files from the source file on YouTube.

    Args:
        dir_song (Path): Path to the song directory to save the audio files.
        original (str): ID of the original audio file on YouTube.
        pianos (List[str]): IDs of the piano audio files on YouTube.
    """
    dir_song_piano = dir_song / DIR_NAME_PIANO
    dir_song_piano.mkdir(exist_ok=True, parents=True)

    ids = [original] + pianos
    urls = [f"https://www.youtube.com/watch?v={id}" for id in ids]
    ydl_opts = {
        "outtmpl": f"{dir_song}/%(id)s.%(ext)s",
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "ignoreerrors": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
    print()

    for piano in pianos:
        piano_wav = dir_song / f"{piano}.wav"
        piano_wav.rename(dir_song_piano / f"{piano}.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download the audio files from the source file on YouTube.")
    parser.add_argument("--path_src", type=str, default=None, help="Path to the source file. Defaults to CONFIG.PATH.SRC.")
    args = parser.parse_args()
    main(args)
