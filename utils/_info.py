from pathlib import Path
import json
from typing import Dict, Any

from ._config import (
    config,
    Config as CustomDict
)


class Info:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            with open(path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = {}
            with open(path, "w") as f:
                json.dump(self.data, f)

    def __getitem__(self, key: str):
        return CustomDict(self.data[key])

    def set(self, id: str, key: str, value: Any, save: bool = True):
        if id not in self.data:
            self.data[id] = {}
        self.data[id][key] = value
        if save:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)

    def update(self, id: str, values: Dict, save: bool = True):
        if id not in self.data:
            self.data[id] = {}
        self.data[id].update(values)
        if save:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)


root = Path(__file__).resolve().parent.parent
info = Info(root / config.path.info)
