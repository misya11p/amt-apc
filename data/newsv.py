import argparse
from pathlib import Path
import json

import numpy as np
import pretty_midi
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


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
PATH_STYLE_VECTOR = "data/style_vector.json"

BIN_VEL = np.arange(1, N_VELOCITY)
BIN_PITCH = np.arange(PITCH_MIN, PITCH_MAX + 1)


def main(args):
    dir_dataset = Path(args.path_dataset)
    dir_input = dir_dataset / DIR_NAME_SYNCED

    pianos = list(dir_input.glob("*/piano/*.mid"))
    pianos = sorted(pianos)

    sum_dist_vel = np.zeros(N_VELOCITY - 1)
    sum_dist_pitch = np.zeros(NUM_PITCH)
    all_onset_rate = []
    all_hold_rate = []
    style_raw = {}
    for piano in tqdm(pianos, desc="Extract style vectors"):
        midi = pretty_midi.PrettyMIDI(str(piano))
        if not midi.instruments:
            continue
        piano_roll = midi.get_piano_roll(int(SR / HOP_LENGTH))
        n_notes = get_n_notes(piano_roll)
        if n_notes < 500:
            continue
        piano_roll = piano_roll[PITCH_MIN:PITCH_MAX + 1]
        dist_vel, dist_pitch, onset_rate, hold_rate = get_style(piano_roll)
        style_raw[piano.stem] = {
            "dist_vel": dist_vel,
            "dist_pitch": dist_pitch,
            "onset_rate": onset_rate,
            "hold_rate": hold_rate
        }
        sum_dist_vel += dist_vel
        sum_dist_pitch += dist_pitch
        all_onset_rate.append(onset_rate)
        all_hold_rate.append(hold_rate)

    mean_vel = np.average(BIN_VEL, weights=sum_dist_vel)
    mean_pitch = np.average(BIN_PITCH, weights=sum_dist_pitch)
    mean_onset_rate = np.mean(all_onset_rate)
    mean_hold_rate = np.mean(all_hold_rate)
    std_vel = np.sqrt(np.average((BIN_VEL - mean_vel) ** 2, weights=sum_dist_vel))
    std_pitch = np.sqrt(np.average((BIN_PITCH - mean_pitch) ** 2, weights=sum_dist_pitch))
    std_onset_rate = np.std(all_onset_rate)
    std_hold_rate = np.std(all_hold_rate)
    params = {
        "mean_vel": mean_vel,
        "mean_pitch": mean_pitch,
        "mean_onset_rate": mean_onset_rate,
        "mean_hold_rate": mean_hold_rate,
        "std_vel": std_vel,
        "std_pitch": std_pitch,
        "std_onset_rate": std_onset_rate,
        "std_hold_rate": std_hold_rate
    }
    bin_vel_norm = (BIN_VEL - mean_vel) / std_vel
    bin_pitch_norm = (BIN_PITCH - mean_pitch) / std_pitch

    style_vector = {}
    for id_piano, style in tqdm(style_raw.items(), desc="Normalize style vectors"):
        dist_vsl = style["dist_vel"]
        dist_pitch = style["dist_pitch"]
        onset_rate = style["onset_rate"]
        hold_rate = style["hold_rate"]

        _mean_vel = np.average(bin_vel_norm, weights=dist_vsl)
        _std_vel = np.average((bin_vel_norm - _mean_vel) ** 2, weights=dist_vsl)

        pitches = (np.arange(NUM_PITCH) - mean_pitch) / std_pitch
        pitches = sum([[p] * n for p, n in zip(bin_pitch_norm, dist_pitch)], [])
        pitches = np.array(pitches)
        _means_pitch, _stds_pitch = get_gmm_params(pitches)

        _mean_onset_rate = (onset_rate - mean_onset_rate) / std_onset_rate
        _mean_hold_rate = (hold_rate - mean_hold_rate) / std_hold_rate

        sv = sum([
            _means_pitch.tolist(),
            _stds_pitch.tolist(),
            [_mean_vel, _std_vel, _mean_onset_rate, _mean_hold_rate]
        ], [])
        style_vector[id_piano] = sv

    out = {
        "style_vector": style_vector,
        "params": params
    }
    with open(PATH_STYLE_VECTOR, "w") as f:
        json.dump(out, f, indent=2)


def get_n_notes(piano_roll):
    onset = np.diff(piano_roll, axis=-1) > 0
    n_notes = onset.sum()
    return n_notes


def get_style(piano_roll):
    n_frames = piano_roll.shape[1]
    onset = np.diff(piano_roll, axis=-1) > 0
    onset = np.pad(onset, ((0, 0), (1, 0)))
    velocity = piano_roll[onset]
    dist_vel = [int((velocity == v).sum()) for v in range(1, N_VELOCITY)]
    dist_pitch = [(np.diff(onset[p]) > 0).sum() for p in range(NUM_PITCH)]
    onset_rate = onset.sum() / n_frames
    hold_rate = piano_roll.astype(bool).sum() / n_frames
    return np.array(dist_vel), np.array(dist_pitch), onset_rate, hold_rate


def get_gmm_params(x):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(x.reshape(-1, 1))
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    if means[0] > means[1]:
        means = means[::-1]
        stds = stds[::-1]
    return means, stds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_dataset", type=str, default="./dataset/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)