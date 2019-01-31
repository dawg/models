import os
import threading
import logme
import tqdm
import zipfile
import boto3
import botocore
import stempeg
import torch
import numpy as np

from vusic.utils.transforms import STFT
from vusic.utils.separation_settings import preprocess_settings, stft_info
from vusic.utils import Downloader


class Set:
    TRAIN = "train"
    TEST = "test"


class Stem:
    MIX = (0,)
    DRUMS = (1,)
    BASS = (2,)
    OTHER = (3,)
    VOCALS = (4,)


def write_stem_pt(
    dst: str,
    fname: str,
    stem: object,
    is_stft: bool = False,
    window_size: int = 4046,
    hop_size: int = 2048,
):
    """
    Desc:
        writes a stem as a .pth to dst with fname as the file name

    Args:
        dst (string): download directory

        fname (string): file name of the stem

        stem (object): numpy array containing stem
    """

    fname += ".pth"

    if is_stft:
        mix = torch.from_numpy(stem[Stem.MIX, :, :].astype(np.float32))
        vocals = torch.from_numpy(stem[Stem.VOCALS, :, :].astype(np.float32))

        stft = STFT(window_size=window_size, hop_size=hop_size)

        mix = mix[:, :, 0]
        vocals = vocals[:, :, 0]

        # XXX maybe as an object instead of a tuple?
        fmix = stft.forward(mix)
        fvocals = stft.forward(vocals)

        torch.save(fmix, os.path.join(dst, "mix", fname))
        torch.save(fvocals, os.path.join(dst, "vocals", fname))

    else:
        mix = torch.from_numpy(stem[Stem.MIX, :, :].astype(np.float16))
        vocals = torch.from_numpy(stem[Stem.VOCALS, :, :].astype(np.float16))

        torch.save(mix, os.path.join(dst, "mix", fname))
        torch.save(vocals, os.path.join(dst, "vocals", fname))


@logme.log
def write(
    src: str,
    dst: str,
    is_stft: bool = False,
    window_size: int = 4094,
    hop_size: int = 2048,
    logger: object = None,
):
    """
    Desc: 
        writes the dataset as either a numpy file or a pytorch file

    Args:
        src (string): directory containing raw samples

        dst (string): destination for binary files
        
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
        suffix = ".pth"

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

                write_stem_pt(
                    dst,
                    fname,
                    stem,
                    is_stft=is_stft,
                    window_size=window_size,
                    hop_size=hop_size,
                )

    except Exception:
        logger.info(f"Removing {dst}")
        if os.path.exists(dst):
            os.remove(dst)
        raise


def main():
    downloader = Downloader.from_params(preprocess_settings["downlader"])

    dst = preprocess_settings["pre_dst"]

    downloader.get_dataset(Set.TRAIN, dst)
    downloader.get_dataset(Set.TEST, dst)

    write(
        os.path.join(dst, "train"),
        os.path.join(dst, "pt_f_train"),
        window_size=stft_info["window_size"],
        hop_size=stft_info["hop_size"],
        is_stft=True,
    )
    write(
        os.path.join(dst, "test"),
        os.path.join(dst, "pt_f_test"),
        window_size=stft_info["window_size"],
        hop_size=stft_info["hop_size"],
        is_stft=True,
    )


if __name__ == "__main__":
    main()
