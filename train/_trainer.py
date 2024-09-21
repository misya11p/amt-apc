from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import dlprog

from models import load_model, save_model
from train._loss import loss_fn
from utils import config


DIR_CHECKPOINTS = ROOT / config.path.checkpoints
NAME_FILE_LOG = "log.txt"
PROG_LABELS = ["loss", "F1 avg", "F1 onset", "F1 frame", "F1 velocity"]


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: int | torch.device,
    freq_save: int = 0,
    prog: dlprog.Progress = None,
    file_log: Path = None,
) -> None:
    """
    Training loop in one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        dataloader (DataLoader): DataLoader to use.
        device (int | torch.device): Device to use.
        freq_save (int, optional): Frequency to save the model.
        prog (dlprog.Progress, optional): Progress bar.
        file_log (Path, optional): Path to the log file.
    """
    model.train()

    for i, batch in enumerate(dataloader, 1):
        optimizer.zero_grad()
        spec, sv, onset, offset, frame, velocity = batch
        spec = spec.to(device)
        sv = sv.to(device)
        onset = onset.to(device)
        offset = offset.to(device)
        frame = frame.to(device)
        velocity = velocity.to(device)

        pred = model(spec, sv)
        label = onset, offset, frame, velocity
        loss, f1 = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        if prog is not None:
            prog.update([loss.item(), sum(f1) / 3, *f1])

        if freq_save and (i % freq_save == 0):
            save_model(model, DIR_CHECKPOINTS / "latest.pth")
            loss, f1, f1_onset, f1_frame, f1_velocity = prog.now_values()
            with open(file_log, "a") as f:
                f.write(
                    f"{i}iter, loss: {loss:.2f}, F1 avg: {f1:.2f}, "
                    f"F1 onset: {f1_onset:.2f}, "
                    f"F1 frame: {f1_frame:.2f}, "
                    f"F1 velocity: {f1_velocity:.2f}\n"
                )


class Trainer:
    def __init__(
        self,
        path_model: str,
        dataset: torch.utils.data.Dataset,
        n_gpus: int,
        with_sv: bool,
        no_load: bool,
        freq_save: int,
    ):
        """
        Trainer for calling in DDP.

        Args:
            path_model (str): Path to the base model.
            dataset (torch.utils.data.Dataset): Dataset to use.
            n_gpus (int): Number of GPUs to use.
            with_sv (bool): Whether to use the style vector.
            no_load (bool): Do not use the base model.
            freq_save (int): Frequency to save the model.
        """
        self.path_model = path_model
        self.dataset = dataset
        self.n_gpus = n_gpus
        self.batch_size = config.train.batch_size
        self.n_epochs = config.train.n_epochs
        self.ddp = (n_gpus >= 2)
        self.with_sv = with_sv
        self.no_load = no_load
        self.freq_save = freq_save

    def setup(self, device: int | torch.device) -> None:
        """Setup the model, optimizer, and dataloader."""
        model = load_model(
            path_model=self.path_model,
            device=device,
            amt=True,
            with_sv=self.with_sv,
            no_load=self.no_load,
        )
        if self.ddp:
            dist.init_process_group("nccl", rank=device, world_size=self.n_gpus)
            model = DDP(model, device_ids=[device])
        self.model = torch.compile(model)
        self.optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
        torch.set_float32_matmul_precision("high")
        if self.ddp:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.n_gpus,
                rank=device,
                shuffle=True,
            )
        else:
            self.sampler = None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.train.batch_size,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
        )

    def __call__(self, device: int | torch.device) -> None:
        """Training loop."""
        self.setup(device)

        is_parent = (not self.ddp) or (device == 0)
        if is_parent:
            date = datetime.now().strftime("%Y-%m%d-%H%M%S")
            dir_checkpoint = DIR_CHECKPOINTS / date
            dir_checkpoint.mkdir(parents=True)
            file_log = dir_checkpoint / NAME_FILE_LOG
            prog = dlprog.train_progress(width=20, label=PROG_LABELS)
            prog.start(n_epochs=self.n_epochs, n_iter=len(self.dataloader))
        else:
            prog = None
            file_log = None

        for n in range(self.n_epochs):
            if self.ddp:
                self.sampler.set_epoch(n)
            train(
                model=self.model,
                optimizer=self.optimizer,
                dataloader=self.dataloader,
                device=device,
                freq_save=self.freq_save if is_parent else 0,
                prog=prog,
                file_log=file_log,
            )

            if is_parent:
                loss, f1, f1_onset, f1_frame, f1_velocity = prog.values[-1]
                path_pc_epoch = dir_checkpoint / f"{n + 1}.pth"
                save_model(self.model, path_pc_epoch)
                with open(file_log, "a") as f:
                    time = datetime.now().strftime("%Y/%m/%d %H:%M")
                    f.write(
                        f"{time}, epoch {n + 1} Finished, "
                        f"loss: {loss:.2f}, F1 avg: {f1:.2f}, "
                        f"F1 onset: {f1_onset:.2f}, "
                        f"F1 frame: {f1_frame:.2f}, "
                        f"F1 velocity: {f1_velocity:.2f}\n"
                    )
