import glob
import os
import re
import torch
import logme
import tqdm
import zipfile
import boto3
import torchaudio
import logger

from vusic.utils.downloader import Downloader
from magenta.music import audio_io
from magenta.music import midi_io
from vusic.utils.transcription_settings import preprocess_settings

test_dirs = ["ENSTDkCl/MUS", "ENSTDkAm/MUS"]
train_dirs = [
    "AkPnBcht/MUS",
    "AkPnBsdf/MUS",
    "AkPnCGdD/MUS",
    "AkPnStgb/MUS",
    "SptkBGAm/MUS",
    "SptkBGCl/MUS",
    "StbgTGd2/MUS",
]


def generate_training_set(unzipped_dir: str):
    """
    Desc:
        Generate the training tensors

    Args:
        unzipped_dir (string): Directory where unzipped data is located
    """
    # TODO Make writer, but for now...
    dst = os.path.expanduser("~")
    dst = os.path.join(dst, "storage", "transcription")
    if not os.path.isdir(dst):
        print(f"Creating {dst}")
        os.mkdir(dst)
        dst = os.path.join(dst,"training")
        os.mkdir(dst)

    train_file_pairs = []
    for d in train_dirs:
        # TODO define and point to directories
        path = os.path.join(unzipped_dir, d)
        path = os.path.join(path, "*.wav")
        wav_files = glob.glob(path)

        # find mid files
        for f in wav_files:
            base_name_root, _ = os.path.splitext(f)
            mid_file = base_name_root + ".mid"
            train_file_pairs.append((f, mid_file))


    


def main():
    downloader = Downloader.from_params(preprocess_settings["downlader"])

    dst = preprocess_settings["pre_dst"]

    downloader.get_dataset()


if __name__ == "__main__":
    main()