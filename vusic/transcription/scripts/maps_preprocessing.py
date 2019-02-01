import glob
import os
import re
import torch
import logme
import tqdm
import zipfile
import boto3

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


def generate_training_set(exclude_ids):
    """
    Desc:
        Generate the training tensors

    Args:
        exclude_ids (string): Desc
    """
    train_file_pairs = []
    for d in train_dirs:
        # TODO define and point to directories
        path = os.path.join(input_dir, directory)
        path = os.path.join(path, "*.wav")
        wav_files = glob.glob(path)


def main():
    downloader = Downloader.from_params(preprocess_settings["downlader"])

    dst = preprocess_settings["pre_dst"]

    downloader.get_dataset()


if __name__ == "__main__":
    main()
