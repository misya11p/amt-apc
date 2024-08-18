import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dlprog import train_progress

from models import load_model, save_model
from _utils import loss_fn


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)
PATH_PC = CONFIG["default"]["pc"]
SV_DIM = CONFIG["model"]["sv_dim"]
DIR_CHECKPOINTS = Path("models/params/checkpoints/")
FILE_NAME_LOG = "log.txt"
JST = timezone(timedelta(hours=+9), "JST")


def train(
    model,
    optimizer,
    dataloader,
    device,
    freq_save=0,
    prog=None,
    file_log=None,
):
    model.train()

    all_loss = 0
    for i, batch in enumerate(dataloader, 1):
        optimizer.zero_grad()
        spec, sv, onset, offset, mpe, velocity = batch
        spec = spec.to(device)
        sv = sv.to(device)
        onset = onset.to(device)
        offset = offset.to(device)
        mpe = mpe.to(device)
        velocity = velocity.to(device)

        pred = model(spec, sv)
        label = onset, offset, mpe, velocity
        loss, f1 = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        all_loss += loss.item()

        if prog is not None:
            prog.update([loss.item(), sum(f1) / 3, *f1])

        if freq_save and (i % freq_save == 0):
            save_model(model, PATH_PC)
            loss, f1, _, _, _ = prog.now_values()
            with open(file_log, "a") as f:
                f.write(f"{i}, loss: {loss}, f1: {f1}\n")

    loss = all_loss / i
    return loss


class Trainer:
    def __init__(self, dataset, n_gpus, batch_size, n_epochs, with_sv):
        self.dataset = dataset
        self.n_gpus = n_gpus
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ddp = (n_gpus > 1)
        self.with_sv = with_sv

    def setup(self, device):
        sv_dim = SV_DIM if self.with_sv else 0
        model = load_model(device=device, amt=True, sv_dim=sv_dim)
        model = model.to(device)
        if self.ddp:
            dist.init_process_group("nccl", rank=device, world_size=self.n_gpus)
            model = DDP(model, device_ids=[device])
        self.model = torch.compile(model)
        self.optimizer = optim.Adam(model.parameters(), lr=2.5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        torch.set_float32_matmul_precision("high")
        if self.ddp:
            self.sampler = DistributedSampler(
                self.dataset, num_replicas=self.n_gpus, rank=device, shuffle=True
            )
        else:
            self.sampler = None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
        )

    def __call__(self, device):
        self.setup(device)

        is_parent = (not self.ddp) or (device == 0)
        if is_parent:
            date = datetime.now(JST).strftime("%Y-%m%d-%H%M%S")
            dir_checkpoint = DIR_CHECKPOINTS / date
            dir_checkpoint.mkdir()
            file_log = dir_checkpoint / FILE_NAME_LOG
            prog = train_progress(width=20, label=["loss", "f1", "f1_onset", "f1_mpe", "f1_velocity"])
            prog.start(n_epochs=self.n_epochs, n_iter=len(self.dataloader))
        else:
            prog = None
            file_log = None

        for n in range(self.n_epochs):
            if self.ddp:
                self.sampler.set_epoch(n)
            loss = train(
                model=self.model,
                optimizer=self.optimizer,
                dataloader=self.dataloader,
                device=device,
                freq_save=100 if is_parent else 0,
                prog=prog,
                file_log=file_log,
            )
            self.scheduler.step(loss)

            if is_parent:
                loss, f1 = prog.values[-1]
                path_pc_epoch = dir_checkpoint / f"{n + 1}.pth"
                save_model(self.model, path_pc_epoch)
                with open(file_log, "a") as f:
                    time = datetime.now(JST).strftime("%Y/%m/%d %H:%M")
                    f.write(f"{time}, epoch {n} finished, loss: {loss}, f1: {f1}\n")
