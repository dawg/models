# XXX this file could very well be trash but it could also be useful

import numpy as np
import subprocess as sp
import os
import json
import re
import warnings
import tempfile as tmp
import soundfile as sf
import argparse
import stempeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indirectory")
    args = parser.parse_args()

    # get all of the filenames
    filenames = sp.check_output(["ls {}/*.mp4".format(args.indirectory)], shell=True)

    # create a location for the resulting files
    sp.call("mkdir mixtures sources", shell=True)

    for filename in filenames.splitlines():

        name = filename[len(args.indirectory) : len(filename) - len(".stem.mp4")]
        name = re.sub(r"\W+", "", name.decode())

        if sp.call("mkdir mixtures/{} sources/{}".format(name, name), shell=True) != 0:

            stem, rate = stempeg.read_stems(filename)

            sf.write(
                "mixtures/{}/mixture.wav".format(name),
                np.asarray(stem[0, :, :]),
                rate,
                subtype="PCM_16",
            )
            sf.write(
                "sources/{}/drums.wav".format(name),
                np.asarray(stem[1, :, :]),
                rate,
                subtype="PCM_16",
            )
            sf.write(
                "sources/{}/bass.wav".format(name),
                np.asarray(stem[2, :, :]),
                rate,
                subtype="PCM_16",
            )
            sf.write(
                "sources/{}/other.wav".format(name),
                np.asarray(stem[3, :, :]),
                rate,
                subtype="PCM_16",
            )
            sf.write(
                "sources/{}/vocals.wav".format(name),
                np.asarray(stem[4, :, :]),
                rate,
                subtype="PCM_16",
            )


if __name__ == "__main__":
    main()
