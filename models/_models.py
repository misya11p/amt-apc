import json
from collections import OrderedDict

import torch
import torch.nn as nn
from collections import OrderedDict

from .hFT_Transformer.amt import AMT
from .hFT_Transformer.model_spec2midi import (
    Encoder_SPEC2MIDI as Encoder,
    Decoder_SPEC2MIDI as Decoder,
    Model_SPEC2MIDI as BaseSpec2MIDI,
)


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)


class Pipeline(AMT):
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        amt: bool = False,
        sv_dim: int = 0,
        path_model: str | None = None,
        skip_load_model: bool = False,
    ):
        """
        Pipeline for converting audio to MIDI. Contains some methods for
        converting audio to MIDI, models, and configurations.

        Args:
            device (torch.device, optional):
                Device to use for the model. Defaults to
                torch.device("cuda" if torch.cuda.is_available() else "cpu").
            amt (bool, optional):
                Whether to use the AMT model.
                Defaults to False (use the cover model).
            path_model (str, optional):
                Path to the model. Defaults to None.
            skip_load_model (bool, optional):
                Whether to skip loading the model. Defaults to False.
        """
        self.device = device
        self.sv_dim = sv_dim
        if skip_load_model:
            self.model = None
        else:
            self.model = load_model(
                device=self.device,
                amt=amt,
                path_model=path_model,
                sv_dim=sv_dim,
            )
        self.config = CONFIG["data"]

    def wav2midi(
        self,
        path_input: str,
        path_output: str,
        sv: None | torch.Tensor = None,
        thred_onset: float = 0.5,
        thred_offset: float = 0.5,
        thred_mpe: float = 0.5,
    ):
        """
        Convert audio to MIDI.

        Args:
            path_input (str): Path to the input audio file.
            path_output (str): Path to the output MIDI file.
            sv (None | torch.Tensor, optional): Style vector.
        """
        feature = self.wav2feature(path_input)
        _, _, _, _, onset, offset, mpe, velocity = self.transcript(feature, sv)
        note = self.mpe2note(
            onset,
            offset,
            mpe,
            velocity,
            thred_onset=thred_onset,
            thred_offset=thred_offset,
            thred_mpe=thred_mpe,
        )
        self.note2midi(note, path_output)


class Spec2MIDI(BaseSpec2MIDI):
    def __init__(self, encoder, decoder, sv_dim: int = 0):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        delattr(self, "encoder_spec2midi")
        delattr(self, "decoder_spec2midi")
        self.sv_dim = sv_dim
        if sv_dim:
            hidden_size = encoder.hid_dim
            self.fc_sv = nn.Linear(sv_dim, hidden_size)

    def forward(self, x, sv: None | torch.Tensor = None):
        h = self.encode(x, sv) # (batch_size, n_frames, hidden_size)
        y = self.decode(h)
        return y

    def encode(self, x, sv=None):
        h = self.encoder(x)
        if self.sv_dim and (sv is not None):
            sv = self.fc_sv(sv) # (batch_size, hidden_size)
            _, n_frames, n_bin, _ = h.shape
            sv = sv.unsqueeze(1).unsqueeze(2)
            sv = sv.repeat(1, n_frames, n_bin, 1)
            h = h + sv
        return h

    def decode(self, h):
        onset_f, offset_f, mpe_f, velocity_f, attention, \
        onset_t, offset_t, mpe_t, velocity_t = self.decoder(h)
        return (
            onset_f, offset_f, mpe_f, velocity_f, attention,
            onset_t, offset_t, mpe_t, velocity_t
        )


def load_model(
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    amt: bool = False,
    path_model: str | None = None,
    sv_dim: int = 0,
    no_load: bool = False,
) -> Spec2MIDI:
    """
    Load the model according to models/config.json.

    Args:
        device (torch.device, optional):
            Device to use for the model. Defaults to
            torch.device("cuda" if torch.cuda.is_available() else "cpu").
        amt (bool, optional):
            Whether to use the AMT model.
            Defaults to False (use the cover model).
        path_model (str, optional):
            Path to the model. Defaults to None.
        no_load (bool, optional):
            Whether to skip loading the model parameters. Defaults to False.
    Returns:
        Spec2MIDI: Model.
    """
    if amt:
        path_model = path_model or CONFIG["default"]["amt"]
    else:
        path_model = path_model or CONFIG["default"]["pc"]

    encoder = Encoder(
        n_margin=CONFIG["data"]["input"]["margin_b"],
        n_frame=CONFIG["data"]["input"]["num_frame"],
        n_bin=CONFIG["data"]["feature"]["n_bins"],
        cnn_channel=CONFIG["model"]["cnn"]["channel"],
        cnn_kernel=CONFIG["model"]["cnn"]["kernel"],
        hid_dim=CONFIG["model"]["transformer"]["hid_dim"],
        n_layers=CONFIG["model"]["transformer"]["encoder"]["n_layer"],
        n_heads=CONFIG["model"]["transformer"]["encoder"]["n_head"],
        pf_dim=CONFIG["model"]["transformer"]["pf_dim"],
        dropout=CONFIG["model"]["training"]["dropout"],
        device=device,
    )
    decoder = Decoder(
        n_frame=CONFIG["data"]["input"]["num_frame"],
        n_bin=CONFIG["data"]["feature"]["n_bins"],
        n_note=CONFIG["data"]["midi"]["num_note"],
        n_velocity=CONFIG["data"]["midi"]["num_velocity"],
        hid_dim=CONFIG["model"]["transformer"]["hid_dim"],
        n_layers=CONFIG["model"]["transformer"]["decoder"]["n_layer"],
        n_heads=CONFIG["model"]["transformer"]["decoder"]["n_head"],
        pf_dim=CONFIG["model"]["transformer"]["pf_dim"],
        dropout=CONFIG["model"]["training"]["dropout"],
        device=device,
    )
    model = Spec2MIDI(encoder, decoder, sv_dim=sv_dim)
    if not no_load:
        state_dict = torch.load(path_model, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def save_model(model: nn.Module, path: str):
    """
    Save the model.

    Args:
        model (nn.Module): Model to save.
        path (str): Path to save the model.
    """
    state_dict = model.state_dict()
    correct_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "").replace("module.", "")
        correct_state_dict[key] = value

    torch.save(correct_state_dict, path)
