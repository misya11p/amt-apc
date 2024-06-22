# >>> ------------------------------------------------------------------
# This code is based on the code from synctoolbox demo notebook:
# https://github.com/meinardmueller/synctoolbox/blob/master/sync_audio_audio_full.ipynb


import os
from contextlib import redirect_stdout

import numpy as np
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import make_path_strictly_monotonic
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning
import pytsmod


FEATURE_RATE = 50
STEP_WEIGHTS = np.array([1.5, 1.5, 2.0])
THRESHOLD_REC = 10 ** 6


def get_features_from_audio(audio, tuning_offset, sr):
    with redirect_stdout(open(os.devnull, "w")):
        f_pitch = audio_to_pitch_features(
            f_audio=audio,
            Fs=sr,
            tuning_offset=tuning_offset,
            feature_rate=FEATURE_RATE,
            verbose=False,
        )
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        f_pitch_onset = audio_to_pitch_onset_features(
            f_audio=audio,
            Fs=sr,
            tuning_offset=tuning_offset,
            verbose=False,
        )
        f_DLNCO = pitch_onset_features_to_DLNCO(
            f_peaks=f_pitch_onset,
            feature_rate=FEATURE_RATE,
            feature_sequence_length=f_chroma_quantized.shape[1],
            visualize=False,
        )
    return f_chroma_quantized, f_DLNCO


def sync_audio(
    y_source: np.ndarray,
    y_target: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Synchronize the source audio with the target audio.

    Args:
        y_source (np.ndarray): Source audio. (n_samples,)
        y_target (np.ndarray): Target audio. (n_samples,)
        sr (int): Sample rate.

    Returns:
        np.ndarray: Synchronized source audio. (n_samples,)
    """
    tuning_offset_source = estimate_tuning(y_source, sr)
    tuning_offset_target = estimate_tuning(y_target, sr)

    f_chroma_quantized_1, f_DLNCO_1 = get_features_from_audio(
        y_source, tuning_offset_source, sr
    )
    f_chroma_quantized_2, f_DLNCO_2 = get_features_from_audio(
        y_target, tuning_offset_target, sr
    )

    wp = sync_via_mrmsdtw(
        f_chroma1=f_chroma_quantized_1,
        f_onset1=f_DLNCO_1,
        f_chroma2=f_chroma_quantized_2,
        f_onset2=f_DLNCO_2,
        input_feature_rate=FEATURE_RATE,
        step_weights=STEP_WEIGHTS,
        threshold_rec=THRESHOLD_REC,
        verbose=False,
    )
    wp = make_path_strictly_monotonic(wp)

    time_map = wp / FEATURE_RATE * sr
    time_map[0, time_map[0, :] > len(y_source)] = len(y_source) - 1
    time_map[1, time_map[1, :] > len(y_target)] = len(y_target) - 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_hptsm = pytsmod.hptsm(y_source, time_map)
    y_hptsm = np.ravel(y_hptsm)
    return y_hptsm


# ------------------------------------------------------------------ <<<


import argparse
from pathlib import Path
import shutil
import multiprocessing
import warnings
import time


import librosa
import soundfile as sf


DIR_NAME_EXTRACTED = "audio-extracted/"
DIR_NAME_SYNCED = "audio-synced/"
DIR_NAME_PIANO = "piano/"


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_EXTRACTED
    dir_output = dir_dataset / DIR_NAME_SYNCED
    dir_output.mkdir(exist_ok=True)

    sync_args = []
    for song in dir_input.glob("*/"):
        sync_args.append((song, dir_output, args.sr, args.overwrite))

    with multiprocessing.Pool(args.n_processes) as pool:
        pool.starmap(_sync_song, sync_args)


def _sync_song(
    dir_song: str,
    dir_output: str,
    sr: int | None = None,
    overwrite: bool = False
):
    """
    Synchronize the piano audio with the original audio in given song
    directory.

    Args:
        dir_song (str): Path to the song directory.
        dir_output (str): Path to the output directory.
        sr (int | None, optional):
            Sample rate of the audio files. If None, use the default
            sample rate of librosa (22050). Defaults to None.
    """
    dir_output_song = dir_output / dir_song.name

    start_time = time.time()
    orig = next(dir_song.glob("*.wav"))
    orig_new = dir_output_song / orig.name
    flag_load_orig = False

    if overwrite or (not orig_new.exists()):
        dir_output_song.mkdir(exist_ok=True)
        y_orig, sr = librosa.load(str(orig), sr=sr)
        flag_load_orig = True
        shutil.copy(orig, str(orig_new))

    dir_output_song_piano = dir_output_song / DIR_NAME_PIANO
    dir_output_song_piano.mkdir(exist_ok=True)
    for i, piano in enumerate((dir_song / DIR_NAME_PIANO).glob("*.wav"), 1):
        piano_new = dir_output_song_piano / piano.name
        if overwrite or (not piano_new.exists()):
            if not flag_load_orig:
                y_orig, sr = librosa.load(str(orig), sr=sr)
                flag_load_orig = True
            y_piano, _ = librosa.load(str(piano), sr=sr)
            y_piano_synced = sync_audio(y_piano, y_orig, sr)
            sf.write(str(piano_new), y_piano_synced, sr)

    running_time = time.strftime("%Mm%Ss", time.gmtime(time.time() - start_time))
    print(f"Synced ({i} pianos) '{dir_song.name}' in {running_time}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize piano audio with original audio.")
    parser.add_argument("-d", "--path_dataset", type=str, default="../dataset/", help="Path to the datasets directory. Defaults to '../datasets/'.")
    parser.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use. Defaults to 1.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate of the audio files. Defaults to 22050.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    main(args)
