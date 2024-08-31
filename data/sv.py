import argparse
from pathlib import Path
import json

import numpy as np
import pretty_midi
from tqdm import tqdm


DIR_NAME_SYNCED = "synced/"
DIR_NAME_PIANO = "piano/"

with open("models/config.json", "r") as f:
    CONFIG = json.load(f)["data"]
PITCH_MIN = CONFIG["midi"]["note_min"]
PITCH_MAX = CONFIG["midi"]["note_max"]
NUM_PITCH = CONFIG["midi"]["num_note"]
N_VELOCITY = CONFIG["midi"]["num_velocity"]
SR = CONFIG["feature"]["sr"]
HOP_LENGTH = CONFIG["feature"]["hop_sample"]
N_FRAMES = CONFIG["input"]["num_frame"]

PATH_STYLES = "data/.styles.json"
PATH_STYLE_VECTOR = "data/style_vector.json"

BIN_VEL = np.arange(1, N_VELOCITY)
BIN_PITCH = np.arange(PITCH_MIN, PITCH_MAX + 1)


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_synced = dir_dataset / DIR_NAME_SYNCED

    pianos = list(dir_synced.glob("*/piano/*.mid"))
    pianos = sorted(pianos)

    if (not args.overwrite) and Path(PATH_STYLES).exists():
        with open(PATH_STYLES, "r") as f:
            raw_styles = json.load(f)
    else:
        raw_styles, ignore_ids = extract_raw_styles(pianos, min_notes=args.min_notes)
        with open(PATH_STYLES, "w") as f:
            json.dump(raw_styles, f)

    params = estimate_params(raw_styles, ignore_ids)
    style_vectors = create_style_vectors(raw_styles, params)
    out = {
        "style_vector": style_vectors,
        "params": params
    }
    with open(PATH_STYLE_VECTOR, "w") as f:
        json.dump(out, f)


def extract_raw_style(path, min_notes=0):
    midi = pretty_midi.PrettyMIDI(str(path))
    piano_roll = midi.get_piano_roll(int(SR / HOP_LENGTH))
    piano_roll = piano_roll[PITCH_MIN:PITCH_MAX + 1]
    n_frames_midi = piano_roll.shape[1]
    onset = np.diff(piano_roll, axis=1) > 0
    status = 0
    if not onset.any():
        return 2, None
    elif onset.sum() < min_notes:
        status = 1
    onset = np.pad(onset, ((0, 0), (1, 0)))
    velocity = piano_roll[onset]
    dist_vel = [int((velocity == v).sum()) for v in BIN_VEL]
    dist_pitch = [int((np.diff(onset[p]) > 0).sum()) for p in range(NUM_PITCH)]

    onset_rates = []
    for i in range(0, n_frames_midi, N_FRAMES):
        seg_onset = onset[:, i:i + N_FRAMES]
        onset_rate = seg_onset.sum() / N_FRAMES
        onset_rates.append(onset_rate)

    return status, (dist_vel, dist_pitch, onset_rates)


def extract_raw_styles(pianos, min_notes=0):
    raw_styles = {}
    ignore_ids = []
    for piano in tqdm(pianos, desc="Extracting raw styles"):
        status, raw_style = extract_raw_style(piano, min_notes)
        if status == 1:
            ignore_ids.append(piano.stem)
        elif status == 2:
            continue

        raw_styles[piano.stem] = {
            "dist_vel": raw_style[0],
            "dist_pitch": raw_style[1],
            "onset_rates": raw_style[2],
        }
    return raw_styles, ignore_ids


def estimate_params(raw_styles, ignore_ids):
    sum_dist_vel = np.zeros(N_VELOCITY - 1)
    sum_dist_pitch = np.zeros(NUM_PITCH)
    all_onset_rate = []
    for pid, style in raw_styles.items():
        if pid in ignore_ids:
            continue
        sum_dist_vel += style["dist_vel"]
        sum_dist_pitch += style["dist_pitch"]
        all_onset_rate += style["onset_rates"]
    mean_vel = np.average(BIN_VEL, weights=sum_dist_vel)
    mean_pitch = np.average(BIN_PITCH, weights=sum_dist_pitch)
    mean_onset_rate = np.mean(all_onset_rate)
    std_vel = np.sqrt(np.average((BIN_VEL - mean_vel) ** 2, weights=sum_dist_vel))
    std_pitch = np.sqrt(np.average((BIN_PITCH - mean_pitch) ** 2, weights=sum_dist_pitch))
    std_onset_rate = np.std(all_onset_rate)
    params = {
        "mean_vel": mean_vel,
        "mean_pitch": mean_pitch,
        "mean_onset_rate": mean_onset_rate,
        "std_vel": std_vel,
        "std_pitch": std_pitch,
        "std_onset_rate": std_onset_rate,
    }
    return params


BIN_DIST = np.array([-2, -4/3, -2/3, 0, 2/3, 4/3, 2])
def create_style_vectors(raw_styles, params):
    mean_vel = params["mean_vel"]
    mean_pitch = params["mean_pitch"]
    mean_onset_rate = params["mean_onset_rate"]
    std_vel = params["std_vel"]
    std_pitch = params["std_pitch"]
    std_onset_rate = params["std_onset_rate"]

    styles = {}
    for id_piano, style in tqdm(raw_styles.items(), desc="Normalize style vectors"):
        dist_vel = style["dist_vel"]
        dist_pitch = style["dist_pitch"]
        onset_rates = style["onset_rates"]

        # To list
        vels = sum([[v] * n for v, n in zip(BIN_VEL, dist_vel)], [])
        pitches = sum([[p] * n for p, n in zip(BIN_PITCH, dist_pitch)], [])
        vels = np.array(vels)
        pitches = np.array(pitches)

        # Normalize
        vels_norm = (vels - mean_vel) / std_vel
        pitches_norm = (pitches - mean_pitch) / std_pitch
        onset_rates_norm = (onset_rates - mean_onset_rate) / std_onset_rate

        # Digitize
        dist_vel = get_distribution(vels_norm)
        dist_pitch = get_distribution(pitches_norm)
        dist_onset_rate = get_distribution(onset_rates_norm)

        # Concatenate
        style_vector = np.concatenate([
            dist_vel, dist_pitch, dist_onset_rate
        ]).tolist()
        styles[id_piano] = style_vector

    return styles

def get_distribution(data):
    digit = np.digitize(data, BIN_DIST)
    dist = [(digit == v).sum() for v in range(len(BIN_DIST) + 1)]
    dist = np.array(dist)
    dist = dist / dist.sum()
    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min_notes", type=int, default=1000)
    args = parser.parse_args()
    main(args)