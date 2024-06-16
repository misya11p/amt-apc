import json
import torch
from .hFT_Transformer.amt import AMT
from .hFT_Transformer.model_spec2midi import Model_SPEC2MIDI
from .utils import load_model


DEFAULT_ARGS = {
    "f_config": "../corpus/config.json",
    "f_list": "../corpus/MAESTRO-V3/list/test.list",
    "d_cp": "../checkpoint",
    "m": "best_model.pkl",
    "mode": "combination",
    "d_wav": "../corpus/MAESTRO-V3/wav",
    "d_fe": "../corpus/MAESTRO-V3/feature",
    "d_mpe": "result/mpe",
    "d_note": "result/note",
    "thred_mpe": 0.5,
    "thred_onset": 0.5,
    "thred_offset": 0.5,
    "calc_feature": False,
    "calc_transcript": False,
    "n_stride": 0,
    "ablation": False
}
with open("models/config.json", "r") as f:
    CONFIG = json.load(f)


class Pipeline(AMT):
    def __init__(self, amt=False, device=None, encoder_path=None, decoder_path=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = load_model(
            amt=amt,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            device=self.device
        )
        self.batch_size = 1
        self.config = CONFIG["data"]

    def wav2midi(self, input_path, output_path):
        feature = self.wav2feature(input_path)
        _, _, _, _, onset, offset, mpe, velocity = self.transcript(feature)
        note = self.mpe2note(onset, offset, mpe, velocity)
        self.note2midi(note, output_path)


