import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from ChromaCoverId import (
    ChromaFeatures,
    cross_recurrent_plot,
    qmax_measure,
)

from utils import info


def main(args):
    dir_input = Path(args.dir_input)
    covers = list(dir_input.glob("*.wav"))
    covers = sorted(covers)

    no_origs = []
    dists = {}
    for cover in covers:
        orig = info.id2path(cover.stem).raw
        if not orig.exists():
            no_origs.append(cover)
            continue
        dist = get_distance(orig, cover)
        dists[cover.stem] = dist
    write_result(args.path_result, dists, no_origs)


def get_distance(path1, path2):
    chroma1 = ChromaFeatures(str(path1))
    chroma2 = ChromaFeatures(str(path2))
    hpcp1 = chroma1.chroma_hpcp()
    hpcp2 = chroma2.chroma_hpcp()
    crp = cross_recurrent_plot(hpcp1, hpcp2)
    qmax, _ = qmax_measure(crp)
    return qmax


def write_result(path, dists, no_origs):
    sim_avg = sum(dists.values()) / len(dists)
    print(f"Average distance: {sim_avg}")
    with open(path, "w") as f:
        f.write(f"Average distance: {sim_avg}\n\n")
        f.write("Distance per cover:\n")
        for cover, dist in dists.items():
            f.write(f"  {cover}: {dist}\n")
        f.write("\n")
        if no_origs:
            f.write("No original found for covers:\n")
            for cover in no_origs:
                f.write(f"  {cover}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate cover similarity using qmax measure.")
    parser.add_argument("--dir_input", type=str, default="ROOT/eval/data/", help="Directory containing cover WAV files.")
    parser.add_argument("--path_result", type=str, default="ROOT/eval/qmax.txt", help="Path to save the result.")
    args = parser.parse_args()
    main(args)
