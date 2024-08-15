import torch
import torch.nn as nn
from sklearn.metrics import f1_score


THRESHOLD = 0.5
BCE_LOSS = nn.BCELoss()
CE_LOSS = nn.CrossEntropyLoss()

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
    f1_avg = (f1_onset + f1_mpe + f1_velocity) / 3
    return f1_avg


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

    # select only the incorrect predictions
    onset_idx = ((onset_pred > THRESHOLD) != onset_label)
    onset_pred = onset_pred[onset_idx]
    onset_label = onset_label[onset_idx]

    velocity_idx = (velocity_pred.argmax(dim=-1).bool() != velocity_label.bool())
    velocity_pred = velocity_pred[velocity_idx]
    velocity_label = velocity_label[velocity_idx]

    # calculate loss
    loss_onset = BCE_LOSS(onset_pred, onset_label)
    loss_mpe = BCE_LOSS(mpe_pred, mpe_label)
    loss_velocity = CE_LOSS(velocity_pred, velocity_label)
    loss = (loss_onset + loss_mpe + loss_velocity) / 3

    return loss, f1