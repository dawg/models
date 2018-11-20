import os
import argparse
import threading
import logme
import tqdm
import zipfile
import boto3
import botocore
import stempeg
import torch
import numpy as np
from boto3.session import Session

ACCESS_KEY = "xxx"
SECRET_KEY = "xxx"
BUCKET_NAME = "vuesic-musdbd18"
OBJECT = "musdb18.zip"
HOME = os.path.expanduser("~")
DST = os.path.join(HOME, "storage", "separation")


class Set:
    TRAIN = "train"
    TEST = "test"


class Stem:
    MIX = (0,)
    DRUMS = (1,)
    BASS = (2,)
    OTHER = (3,)
    VOCALS = (4,)


class CallbackProgressBar(object):
    def __init__(self, total: int, unit: str = None):
        """
        Args:
            total (int): Total number of iterations

            unit (string, optional): Unit with respect to each iteration
        """
        self.pbar = tqdm.tqdm(total=total, unit=unit)
        self.lock = threading.Lock()

    def __call__(self, update: int):
        """
        Args:
            update (int): Iterations completed since last update
        """
        with self.lock:
            self.pbar.update(update)


@logme.log
def download_dataset(key: str, dst: str, logger=None):
    """
    Args:
        key (string): filename to be retrieved from our bucket

        dst (string): download directory

        logger (object, optional): logger object (taken care of by decorator)
    """
    # Download dst
    path = os.path.join(dst, key)

    if not os.path.isfile(path):

        logger.info("Download started")

        session = Session(
            aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
        )

        bucket = session.resource("s3").Bucket(BUCKET_NAME)

        pbar = CallbackProgressBar(
            bucket.Object(key).get()["ContentLength"], unit="bytes"
        )

        try:
            bucket.download_file(key, path, Callback=pbar)
        except Exception:
            logger.info(f"Failed to download {key}")
            raise

    return path


@logme.log
def get_dataset(set: str, path: str = None, logger: object = None):
    """
    Args:
        set (string): filename to be retrieved from our bucket

        path (string, optional): path to musdb18.zip if it has already been downloaded

        logger (object, optional): logger
    """
    dst = DST

    if not os.path.exists(dst):
        logger.info(f"Making {dst}")
        os.makedirs(dst)

    if path == None:
        path = download_dataset(OBJECT, DST)

    with zipfile.ZipFile(path, "r") as z:

        logger.info(f"Extracting files from {path}")
        for fname in tqdm.tqdm(z.namelist(), unit="Ex"):
            if fname.startswith(set) and not os.path.exists(os.path.join(dst, fname)):
                z.extractall(path=dst, members=[fname])

    return


def write_stem_np(dst: str, fname: str, stem: object):
    """
    Args:
        dst (string): download directory

        fname (string): file name of the stem

        stem (object): numpy array containing stem
    """

    mix = torch.from_numpy(stem[Stem.MIX, :, :].astype(np.float16))
    vocals = torch.from_numpy(stem[Stem.VOCALS, :, :].astype(np.float16))

    np.save(os.path.join(dst, "mix", fname), mix)
    np.save(os.path.join(dst, "vocals", fname), vocals)


def write_stem_pt(dst: str, fname: str, stem: object):
    """
    Args:
        dst (string): download directory

        fname (string): file name of the stem

        stem (object): numpy array containing stem
    """

    mix = torch.from_numpy(stem[Stem.MIX, :, :].astype(np.float16))
    vocals = torch.from_numpy(stem[Stem.VOCALS, :, :].astype(np.float16))

    fname += ".pth"

    torch.save(mix, os.path.join(dst, "mix", fname))
    torch.save(vocals, os.path.join(dst, "vocals", fname))


@logme.log
def write(src: str, dst: str, asnp: bool = False, logger: object = None):
    """
    Args:
        src (string): directory containing raw samples

        dst (string): destination for binary files

        asnp (bool): save as numpy arrays? Will save as pytorch tensor by default
        
        logger (object, optional): logger
    """

    if not os.path.isdir(src):
        raise FileNotFoundError(f"{src} does not exist!")

    if not os.path.isdir(dst):
        logger.info(f"Creating {dst}")
        os.mkdir(dst)

    vocaldst = os.path.join(dst, "vocals")

    if not os.path.exists(vocaldst):
        os.mkdir(vocaldst)

    mixdst = os.path.join(dst, "mix")

    if not os.path.exists(mixdst):
        os.mkdir(mixdst)

    try:
        logger.info(f"Reading examples from {src}")
        suffix = ".npy" if asnp else ".pth"

        for fname in tqdm.tqdm(os.listdir(src), unit="Ex"):
            if not fname.endswith("stem.mp4"):
                continue

            sname = os.path.join(src, fname)

            if not os.path.exists(sname):
                raise FileNotFoundError(f"{sname} not found")

            if not os.path.exists(
                os.path.join(dst, "vocals", fname + suffix)
            ) and not os.path.exists(os.path.join(dst, "mix", fname + suffix)):

                stem, rate = stempeg.read_stems(sname)

                if asnp:
                    write_stem_np(dst, fname, stem)
                else:
                    write_stem_pt(dst, fname, stem)

    except Exception:
        logger.info(f"Removing {dst}")
        if os.path.exists(dst):
            os.remove(dst)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the musdb18 dataset into a tensorflow record."
    )

    parser.add_argument(
        "--local-path", help="Path to musdb18.zip if it has already been downloaded"
    )
    args = parser.parse_args()

    path = args.local_path

    get_dataset(Set.TRAIN, path)
    get_dataset(Set.TEST, path)

    write(os.path.join(DST, "train"), os.path.join(DST, "pt_train"))
    write(os.path.join(DST, "test"), os.path.join(DST, "pt_test"))


if __name__ == "__main__":
    main()
