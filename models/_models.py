import json
import torch
from .hFT_Transformer.amt import AMT
from ._utils import load_model


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)


class Pipeline(AMT):
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        amt: bool = False,
        encoder_path: str | None = None,
        decoder_path: str | None = None,
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
                encoder_path=encoder_path,
                decoder_path=decoder_path,
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
