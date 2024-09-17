from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from utils import config


THRESHOLD = 0.5
BCE_LOSS = nn.BCELoss()
CE_LOSS = nn.CrossEntropyLoss()
BETA = config.train.beta
THETA_ONSET = config.train.theta_onset
THETA_FRAME = config.train.theta_frame
THETA_VELOCITY = config.train.theta_velocity


def f1_fn(
    onset_pred,
    frame_pred,
    velocity_pred,
    onset_label,
    frame_label,
    velocity_label
):
    onset_pred = onset_pred.cpu().numpy().flatten()
    frame_pred = frame_pred.cpu().numpy().flatten()
    velocity_pred = velocity_pred.cpu().numpy().flatten()
    onset_label = onset_label.cpu().numpy().flatten()
    frame_label = frame_label.cpu().numpy().flatten()
    velocity_label = velocity_label.cpu().numpy().flatten()
    f1_onset = f1_score(onset_label, onset_pred, zero_division=1)
    f1_frame = f1_score(frame_label, frame_pred, zero_division=1)
    f1_velocity = f1_score(velocity_label, velocity_pred, zero_division=1)
    return f1_onset, f1_frame, f1_velocity


def select(label, prob=0.):
    idx_pos = (label > 0)
    shifted_p = torch.roll(idx_pos, shifts=1, dims=-1)
    shifted_n = torch.roll(idx_pos, shifts=-1, dims=-1)
    idx_rand = torch.rand(idx_pos.shape).to(idx_pos.device) < prob
    idx = idx_pos | shifted_p | shifted_n | idx_rand
    return idx


def loss_fn(pred, label, beta=0.75):
    # unpack
    onset_pred_f, offset_pred_f, frame_pred_f, velocity_pred_f, _, \
    onset_pred_t, offset_pred_t, frame_pred_t, velocity_pred_t = pred

    onset_label, offset_label, frame_label, velocity_label = label
    frame_label = frame_label.float()

    with torch.no_grad():
        f1 = f1_fn(
            (onset_pred_t > THRESHOLD),
            (frame_pred_t > THRESHOLD),
            velocity_pred_t.argmax(dim=-1).bool(),
            onset_label.bool(),
            frame_label.bool(),
            velocity_label.bool()
        )

    # select
    onset_idx = select(onset_label, prob=THETA_ONSET)
    onset_pred_f = onset_pred_f[onset_idx]
    onset_pred_t = onset_pred_t[onset_idx]
    onset_label = onset_label[onset_idx]

    frame_idx = select(frame_label, prob=THETA_FRAME)
    frame_pred_f = frame_pred_f[frame_idx]
    frame_pred_t = frame_pred_t[frame_idx]
    frame_label = frame_label[frame_idx]

    velocity_idx = select(velocity_label, prob=THETA_VELOCITY)
    velocity_pred_f = velocity_pred_f[velocity_idx]
    velocity_pred_t = velocity_pred_t[velocity_idx]
    velocity_label = velocity_label[velocity_idx]

    # calculate loss
    loss_onset_f = BCE_LOSS(onset_pred_f, onset_label)
    loss_onset_t = BCE_LOSS(onset_pred_t, onset_label)

    loss_frame_f = BCE_LOSS(frame_pred_f, frame_label)
    loss_frame_t = BCE_LOSS(frame_pred_t, frame_label)

    # 後で直す
    velocity_label = velocity_label.long()
    loss_velocity_f = CE_LOSS(velocity_pred_f, velocity_label)
    loss_velocity_t = CE_LOSS(velocity_pred_t, velocity_label)

    loss_f = (loss_onset_f + loss_frame_f + loss_velocity_f) / 3
    loss_t = (loss_onset_t + loss_frame_t + loss_velocity_t) / 3
    loss = BETA * loss_f + (1 - BETA) * loss_t

    return loss, f1