from pathlib import Path
import json
from typing import Dict


class Config:
    def __init__(self, config: Dict):
        self.config = config

    def __getattr__(self, name: str) -> Dict:
        value = self.config[name]
        if isinstance(value, dict):
            return Config(value)
        else:
            return value

    def __repr__(self) -> str:
        return self.config.__repr__()


dir_this = Path(__file__).parent.parent
file_config = dir_this / "config.json"
with open(file_config, "r") as f:
    config_raw = json.load(f)
config = Config(config_raw)
