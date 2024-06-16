import json
import torch
from .hFT_Transformer.amt import AMT
from .hFT_Transformer.model_spec2midi import (
    Model_SPEC2MIDI as Spec2MIDI,
    Encoder_SPEC2MIDI as Encoder,
    Decoder_SPEC2MIDI as Decoder,
)


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)


def load_model(
    amt: bool = False,
    encoder_path=None,
    decoder_path=None,
    device=torch.device("cpu"),
):
    if amt:
        encoder_path = encoder_path or CONFIG["default"]["amt_encoder"]
    else:
        encoder_path = encoder_path or CONFIG["default"]["pc_encoder"]
    decoder_path = decoder_path or CONFIG["default"]["decoder"]

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
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.to(device)
    decoder.to(device)
    model = Spec2MIDI(encoder, decoder)
    return model
