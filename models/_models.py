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
Z_DIM = CONFIG["z_dim"]


class Pipeline(AMT):
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        amt: bool = False,
        model_path: str | None = None,
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
            encoder_path (str, optional):
                Path to the encoder model.
                Defaults to None (use the default path).
            decoder_path (str, optional):
                Path to the decoder model.
                Defaults to None (use the default path).
            skip_load_model (bool, optional):
                Whether to skip loading the model. Defaults to False.
        """
        self.device = device
        if skip_load_model:
            self.model = None
        else:
            self.model = load_model(
                device=self.device,
                amt=amt,
                model_path=model_path,
            )
        self.config = CONFIG["data"]

    def wav2midi(self, path_input: str, path_output: str):
        """
        Convert audio to MIDI.

        Args:
            path_input (str): Path to the input audio file.
            path_output (str): Path to the output MIDI file.
        """
        feature = self.wav2feature(path_input)
        _, _, _, _, onset, offset, mpe, velocity = self.transcript(feature)
        note = self.mpe2note(onset, offset, mpe, velocity)
        self.note2midi(note, path_output)


class Spec2MIDI(BaseSpec2MIDI):
    def __init__(self, encoder, decoder, z_dim=None):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        delattr(self, "encoder_spec2midi")
        delattr(self, "decoder_spec2midi")
        self.with_z = bool(z_dim)
        if z_dim:
            hidden_size = encoder.hid_dim
            self.fc_z = nn.Linear(hidden_size + z_dim, hidden_size)

    def forward(self, x, z=None):
        h = self.encode(x, z) # (batch_size, n_frames, hidden_size)
        y = self.decode(h)
        return y

    def encode(self, x, z=None):
        h = self.encoder(x)
        if self.with_z:
            _, n_frames, n_bin, _ = h.shape
            z = z.unsqueeze(1).unsqueeze(2)
            z = z.repeat(1, n_frames, n_bin, 1)
            h = torch.cat([h, z], dim=-1) # (batch_size, n_frames, hidden_size + z_dim)
            h = self.fc_z(h) # (batch_size, n_frames, hidden_size)
        return h

    def decode(self, h):
        onset_f, offset_f, mpe_f, velocity_f, _, \
        onset_t, offset_t, mpe_t, velocity_t = self.decoder(h)
        return (
            onset_f, offset_f, mpe_f, velocity_f,
            onset_t, offset_t, mpe_t, velocity_t
        )


def load_model(
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    amt: bool = False,
    model_path: str | None = None,
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
        path_encoder (str, optional):
            Path to the encoder model.
            Defaults to None (use the default path).
        path_decoder (str, optional):
            Path to the decoder model.
            Defaults to None (use the default path).
    Returns:
        Spec2MIDI: Model.
    """
    if amt:
        model_path = model_path or CONFIG["default"]["amt"]
    else:
        model_path = model_path or CONFIG["default"]["pc"]

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
    if amt:
        model = Spec2MIDI(encoder, decoder)
    else:
        model = Spec2MIDI(encoder, decoder, z_dim=Z_DIM)
    if not no_load:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def save_model(model: torch.nn.Module, path: str):
    """
    Save the model.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Path to save the model.
    """
    state_dict = model.state_dict()
    correct_state_dict = OrderedDict()
    for key, value in state_dict.items():
        correct_state_dict[key.replace("_orig_mod.", "")] = value
    torch.save(correct_state_dict, path)
