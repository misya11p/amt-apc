import json

import torch
import torch.nn as nn
from sklearn.metrics import f1_score


THRESHOLD = 0.5
BCE_LOSS = nn.BCELoss()
CE_LOSS = nn.CrossEntropyLoss()
with open("models/config.json", "r") as f:
    CONFIG = json.load(f)
N_VELOCITY = CONFIG["data"]["midi"]["num_velocity"]


def f1_fn(
    onset_pred,
    mpe_pred,
    velocity_pred,
    onset_label,
    mpe_label,
    velocity_label
):
    onset_pred = onset_pred.cpu().numpy().flatten()
    mpe_pred = mpe_pred.cpu().numpy().flatten()
    velocity_pred = velocity_pred.cpu().numpy().flatten()
    onset_label = onset_label.cpu().numpy().flatten()
    mpe_label = mpe_label.cpu().numpy().flatten()
    velocity_label = velocity_label.cpu().numpy().flatten()
    f1_onset = f1_score(onset_label, onset_pred, zero_division=1)
    f1_mpe = f1_score(mpe_label, mpe_pred, zero_division=1)
    f1_velocity = f1_score(velocity_label, velocity_pred, zero_division=1)
    # f1_avg = (f1_onset + f1_mpe + f1_velocity) / 3
    return f1_onset, f1_mpe, f1_velocity


def select(label, thr=0.5, prob=0.3):
    idx = (label > thr)
    shifted_p = torch.roll(idx, shifts=1, dims=-1)
    shifted_n = torch.roll(idx, shifts=-1, dims=-1)
    random_idx = torch.rand(idx.shape).to(idx.device) < prob
    idx = idx | shifted_p | shifted_n | random_idx
    return idx


def loss_fn(pred, label):
    # unpack
    onset_f_pred, offset_f_pred, mpe_f_pred, velocity_f_pred, _, \
    onset_pred, offset_pred, mpe_pred, velocity_pred = pred

    onset_label, offset_label, mpe_label, velocity_label = label

    with torch.no_grad():
        f1 = f1_fn(
            (onset_pred > THRESHOLD),
            (mpe_pred > THRESHOLD),
            velocity_pred.argmax(dim=-1).bool(),
            onset_label.bool(),
            mpe_label.bool(),
            velocity_label.bool()
        )

    # select
    onset_idx = select(onset_label, prob=0.25)
    onset_f_pred = onset_f_pred[onset_idx]
    onset_pred = onset_pred[onset_idx]
    onset_label = onset_label[onset_idx]

    mpe_idx = select(mpe_label, prob=0.3)
    mpe_f_pred = mpe_f_pred[mpe_idx]
    mpe_pred = mpe_pred[mpe_idx]
    mpe_label = mpe_label[mpe_idx]

    velocity_idx = select(velocity_label, prob=0.01)
    velocity_f_pred = velocity_f_pred[velocity_idx]
    velocity_pred = velocity_pred[velocity_idx]
    velocity_label = velocity_label[velocity_idx]

    # calculate loss
    loss_onset_f = BCE_LOSS(onset_f_pred, onset_label)
    loss_onset = BCE_LOSS(onset_pred, onset_label)

    loss_mpe_f = BCE_LOSS(mpe_f_pred, mpe_label)
    loss_mpe = BCE_LOSS(mpe_pred, mpe_label)

    loss_velocity_f = CE_LOSS(velocity_f_pred, velocity_label)
    loss_velocity = CE_LOSS(velocity_pred, velocity_label)

    loss = (
        loss_onset_f + loss_onset +
        loss_mpe_f + loss_mpe +
        loss_velocity_f + loss_velocity
    ) / 6

    return loss, f1