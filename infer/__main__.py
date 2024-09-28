from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch

from models import Pipeline
from data import SVSampler
from utils import config


DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SV_SAMPLER = SVSampler()

# Transcription function
def transcribe(src, output_path, style="level2", path_model=None, device=None):
    """Transcribe the input WAV or YouTube URL to a MIDI file.
    
    Args:
        src (str): Path to the input WAV file or YouTube URL.
        output_path (str): Path to save the output MIDI file.
        style (str): Cover style. Valid values are 'level1', 'level2', and 'level3'.
        path_model (str, optional): Path to the model. Defaults to config path.
        device (torch.device, optional): Device to run the model on. Defaults to cuda if available.
    """
    path_model = path_model or config.path.apc
    device = torch.device(device) if device else DEVICE_DEFAULT
    pipeline = Pipeline(path_model, device)

    if src.startswith("https://"):
        src = download(src)

    sv = SV_SAMPLER.sample(params=style)
    pipeline.wav2midi(src, output_path, sv, silent=False)


def download(url):
    """Download audio from YouTube and return the path to the WAV file."""
    from yt_dlp import YoutubeDL
    ydl_opts = {
        "outtmpl": "_audio.%(ext)s",
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
        ydl.download([url])
    return "_audio.wav"


# Main function to handle argument parsing
def main(args):
    transcribe(
        src=args.input,
        output_path=args.output,
        style=args.style,
        path_model=args.path_model,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input wav file or URL of YouTube video")
    parser.add_argument("-o", "--output", type=str, default="output.mid", help="Path to the output midi file. Defaults to 'output.mid'")
    parser.add_argument("-s", "--style", type=str, default="level2", help="Cover style. Valid values are 'level1', 'level2', and 'level3'. Defaults to 'level2'")
    parser.add_argument("--path_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)
