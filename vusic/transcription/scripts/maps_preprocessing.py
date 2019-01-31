import glob
import os
import re
import torch
import logme
import tqdm
import zipfile
import boto3
import torchaudio


from vusic.utils.downloader import Downloader

# from magenta.music import audio_io
# from magenta.music import midi_io
from vusic.utils.transcription_settings import preprocess_settings

test_dirs = ["MAPS/MAPS_ENSTDkCl_2/ENSTDkCl/MUS", "MAPS/MAPS_ENSTDkAm_2/ENSTDkCl/MUS"]
train_dirs = [
    "MAPS/MAPS_AkPnBcht_2/AkPnBcht/MUS",
    "MAPS/MAPS_AkPnBsdf_2/AkPnBsdf/MUS",
    "MAPS/MAPS_AkPnCGdD_2/AkPnCGdD/MUS",
    "MAPS/MAPS_AkPnStgb_2/AkPnStgb/MUS",
    "MAPS/MAPS_SptkBGAm_2/SptkBGAm/MUS",
    "MAPS/MAPS_SptkBGCl_2/SptkBGCl/MUS",
    "MAPS/MAPS_StbgTGd2/StbgTGd2/MUS",
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
        dst = os.path.join(dst, "training")
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

    downloader = Downloader.from_params(preprocess_settings["downloader"])

    dst = preprocess_settings["pre_dst"]

    for d in train_dirs:
        downloader.get_dataset(d, dst)

    for d in test_dirs:
        downloader.get_dataset(d, dst)


if __name__ == "__main__":
    main()
