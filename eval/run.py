import argparse
from pathlib import Path

from ChromaCoverId import ChromaFeatures, cross_recurrent_plot, dmax_measure


DIR_NAME_SONGS = "songs"
DIR_NAME_COVER_AUDIO = "cover_audio"


def main(args):
    dir_data = Path(args.dir_data)
    dir_cover = dir_data / DIR_NAME_COVER_AUDIO
    dir_songs = dir_data / DIR_NAME_SONGS

    covers = list(dir_cover.glob("*.wav"))
    covers = sorted(covers)

    no_origs = []
    dists = {}

    for cover in covers:
        orig = dir_songs / cover.name
        if not orig.exists():
            no_origs.append(cover)
            continue

        dist = get_distance(orig, cover)
        dists[cover.stem] = dist

    write_result(dists, no_origs, args.path_output)


def get_distance(path1, path2):
    chroma1 = ChromaFeatures(str(path1))
    chroma2 = ChromaFeatures(str(path2))
    hpcp1 = chroma1.chroma_hpcp()
    hpcp2 = chroma2.chroma_hpcp()
    crp = cross_recurrent_plot(hpcp1, hpcp2)
    dmax, _ = dmax_measure(crp)
    return dmax


def write_result(dists, no_origs, path_output):
    sim_avg = sum(dists.values()) / len(dists)
    with open(path_output, "w") as f:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir_data", type=str, default="./eval/data/")
    parser.add_argument("-o", "--path_output", type=str, default="./eval/result.txt")
    args = parser.parse_args()
    main(args)
