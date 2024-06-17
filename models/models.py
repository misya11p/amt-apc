import json
import torch
from .hFT_Transformer.amt import AMT
from .hFT_Transformer.model_spec2midi import Model_SPEC2MIDI
from .utils import load_model


with open("models/config.json", "r") as f:
    CONFIG = json.load(f)


class Pipeline(AMT):
    def __init__(
        self,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        amt=False,
        encoder_path=None,
        decoder_path=None,
    ):
        self.device = device
        self.model = load_model(
            device=self.device,
            amt=amt,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
        )
        self.config = CONFIG["data"]

    def wav2midi(self, input_path, output_path):
        feature = self.wav2feature(input_path)
        _, _, _, _, onset, offset, mpe, velocity = self.transcript(feature)
        note = self.mpe2note(onset, offset, mpe, velocity)
        self.note2midi(note, output_path)
