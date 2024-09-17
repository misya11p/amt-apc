from pathlib import Path
import json
from typing import Dict, Any


class Config(dict):
    def __init__(self, config: Dict):
        super().__init__(config)

    def __getattr__(self, name: str) -> Dict | Any:
        value = self[name]
        if isinstance(value, dict):
            return Config(value)
        else:
            return value


root = Path(__file__).parent.parent
path_config = root / "config.json"
with open(path_config, "r") as f:
    config_json = json.load(f)
config = Config(config_json)
