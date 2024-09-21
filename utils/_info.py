from pathlib import Path
import json
from typing import Dict, Any


from ._config import config, CustomDict


ROOT = Path(__file__).parent.parent
PATH_DATASET = ROOT / config.path.dataset
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
        for id in self.data:
            self.data[id] = CustomDict(self.data[id])
        self._set_id2path()

    def _set_id2path(self):
        id2path = {}
        for id_piano, info in self.data.items():
            id_orig = info["original"]
            title = info["title"]
            if id_orig not in id2path:
                id2path[id_orig] = {
                    "raw": PATH_DATASET / "raw" / title / f"{id_orig}.wav",
                    "synced": {
                        "wav": PATH_DATASET / "synced" / title / f"{id_orig}.wav",
                        "midi": PATH_DATASET / "synced" / title / f"{id_orig}.mid",
                    },
                    "array": PATH_DATASET / "array" / title / f"{id_orig}.npy",
                }
            id2path[id_piano] = {
                "raw": PATH_DATASET / "raw" / title / "piano" / f"{id_piano}.wav",
                "synced": {
                    "wav": PATH_DATASET / "synced" / title / "piano" / f"{id_piano}.wav",
                    "midi": PATH_DATASET / "synced" / title / "piano" / f"{id_piano}.mid",
                },
                "array": PATH_DATASET / "array" / title / "piano" / f"{id_piano}.npz",
            }
        self._id2path = CustomDict(id2path)

    def __getitem__(self, id: str):
        return self.data[id]

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

    def id2path(self, id: str, orig: bool = False):
        if orig:
            return self._id2path[self.piano2orig(id)]
        else:
            return self._id2path[id]

    def get_ids(self, split: str, orig: bool = False):
        ids = [id for id, info in self.data.items() if info["split"] == split]
        if orig:
            ids = list(set([self.piano2orig(id) for id in ids]))
        return ids


root = Path(__file__).resolve().parent.parent
info = Info(root / config.path.info)
