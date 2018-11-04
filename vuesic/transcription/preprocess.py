import os
import tarfile
from collections import Callable
from typing import List

import logme
import requests
import tqdm

from vuesic.transcription import writer


class Set:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{}.jsonwav.tar.gz"

HOME = os.path.expanduser("~")
DST = os.path.join(HOME, "storage", "transcription")


@logme.log
def get_dataset(s, logger):
    if not os.path.exists(DST):
        logger.info("Making {}".format(DST))
        os.makedirs(DST)

    url = URL.format(s)
    dst = os.path.join(DST, os.path.basename(url))

    fname = os.path.basename(url).split(".")[0]
    final_dst = os.path.join(DST, fname)
    if os.path.exists(final_dst):
        logger.info("{} already exists!".format(final_dst))
        return

    logger.info("Downloading {} to {}".format(url, dst))
    if not download(url, dst):
        logger.info("{} already exists!".format(dst))

    src = dst

    def for_each(member: tarfile.TarInfo):
        logger.debug(
            "Extracting {} to {}".format(member.path, os.path.join(DST, member.path))
        )

    logger.info("Extracting {} to {}".format(src, DST))
    if not extract(src, DST, for_each=for_each):
        logger.info("The extraction did not occur as it will not overwrite files!")


def extract(src: str, dst: str, for_each: Callable = None):
    def track_progress(members: List[tarfile.TarInfo]):
        for member in members:
            if for_each:
                for_each(member)
            yield member

    def conflict(member: tarfile.TarInfo):
        return os.path.exists(os.path.join(dst, member.path))

    if not os.path.isdir(dst):
        raise ValueError("{} must be a dir".format(dst))

    with tarfile.open(src, "r:gz") as tar:
        if any(map(conflict, tar.getmembers())):
            return False

        tar.extractall(path=dst, members=track_progress(tar))
        return True


def download(src: str, dst: str):
    """
    Downloads a file to a destination with a progress bar.

    :param src: The source URL.
    :param dst: The full destination.
    :return: False is the file exists, True otherwise.
    """
    if os.path.exists(dst):
        return False

    directory = os.path.dirname(dst)
    if not os.path.exists(directory):
        raise FileNotFoundError("{} does not exist".format(directory))

    with open(dst, "wb") as f:
        response = requests.get(src, stream=True)
        total_length = response.headers.get("content-length")

        if total_length is None:  # no content length header
            f.write(response.content)
            return True

        divider = 2 ** 10

        total_length = int(total_length) // divider
        with tqdm.tqdm(total=total_length, unit="KB") as bar:
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                bar.update(round(len(data) / divider))

        return True


def main():
    get_dataset(Set.VALID)
    get_dataset(Set.TEST)

    writer.write(
        os.path.join(DST, "nsynth-test", "examples.json"),
        os.path.join(DST, "valid.tfrecord"),
    )
    writer.write(
        os.path.join(DST, "nsynth-valid", "examples.json"),
        os.path.join(DST, "train.tfrecord"),
    )


if __name__ == "__main__":
    main()
