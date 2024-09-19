from pathlib import Path
import json
from typing import Dict, Any


from ._config import (
    config,
    Config as CustomDict
)


ROOT = Path(__file__).parent.parent
PATH_MOVIES = ROOT / config.path.src


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

    def export(self):
        movies = {}
        for id, info in self.data.items():
            if not info["include_dataset"]:
                continue

            title = info["title"]
            if title not in movies:
                movies[title] = {
                    "original": info["original"],
                    "pianos": []
                }
            movies[title]["pianos"].append(id)

        with open(PATH_MOVIES, "w") as f:
            json.dump(movies, f, indent=2, ensure_ascii=False)

    def piano2orig(self, id: str):
        return self[id].original

    def is_train(self, id: str):
        return (self[id].split == "train")

    def is_test(self, id: str):
        return (self[id].split == "test")


root = Path(__file__).resolve().parent.parent
info = Info(root / config.path.info)
