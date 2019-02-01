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
import librosa

from vusic.utils import STFT
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


stft = STFT.from_params(stft_info)


def write_stem_pt(dst: str, fname: str, stem: object, is_stft: bool = False):
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
        mix = np.transpose(stem[Stem.MIX, :, :].astype(np.float32))
        vocals = np.transpose(stem[Stem.VOCALS, :, :].astype(np.float32))

        mix = mix[:, :, 0]
        vocals = vocals[:, :, 0]

        mix = librosa.core.to_mono(mix)
        vocals = librosa.core.to_mono(vocals)

        fmix = stft.forward(mix)
        fvocals = stft.forward(vocals)

        fmix = {
            "mg": torch.from_numpy(np.abs(fmix).astype(np.float16)),
            "ph": torch.from_numpy(np.angle(fmix).astype(np.float16)),
        }

        fvocals = {
            "mg": torch.from_numpy(np.abs(fvocals).astype(np.float16)),
            "ph": torch.from_numpy(np.angle(fvocals).astype(np.float16)),
        }

        torch.save(fmix, os.path.join(dst, "mix", fname))
        torch.save(fvocals, os.path.join(dst, "vocals", fname))

    else:
        mix = torch.from_numpy(stem[Stem.MIX, :, :].astype(np.float16))
        vocals = torch.from_numpy(stem[Stem.VOCALS, :, :].astype(np.float16))

        torch.save(mix, os.path.join(dst, "mix", fname))
        torch.save(vocals, os.path.join(dst, "vocals", fname))


@logme.log
def write(src: str, dst: str, is_stft: bool = False, logger: object = None):
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

            checks = not os.path.exists(
                os.path.join(dst, "vocals", fname + suffix)
            ) and not os.path.exists(os.path.join(dst, "mix", fname + suffix))

            if checks:

                stem, rate = stempeg.read_stems(sname)

                write_stem_pt(dst, fname, stem, is_stft=is_stft)

    except Exception:
        logger.info(f"Removing {dst}")
        if os.path.exists(dst):
            os.remove(dst)
        raise


def main():
    downloader = Downloader.from_params(preprocess_settings["downloader"])

    dst = preprocess_settings["pre_dst"]

    downloader.get_dataset(Set.TRAIN, dst)
    downloader.get_dataset(Set.TEST, dst)

    write(os.path.join(dst, "train"), os.path.join(dst, "pt_f_train"), is_stft=True)
    write(os.path.join(dst, "test"), os.path.join(dst, "pt_f_test"), is_stft=True)


if __name__ == "__main__":
    main()
