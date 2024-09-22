from pathlib import Path
import json
from typing import Dict, Any


class CustomDict(dict):
    def __init__(self, config: Dict):
        super().__init__(config)

    def __getattr__(self, name: str) -> Dict | Any:
        value = self[name]
        if isinstance(value, dict):
            return CustomDict(value)
        else:
            return value

    def __getitem__(self, key: Any) -> Any:
        item = super().__getitem__(key)
        if isinstance(item, dict):
            return CustomDict(item)
        else:
            return item


root = Path(__file__).parent.parent
path_config = root / "config.json"
with open(path_config, "r") as f:
    config_json = json.load(f)
config = CustomDict(config_json)
