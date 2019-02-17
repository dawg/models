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

test_dirs = ["ENSTDkCl/MUS", "ENSTDkCl/MUS"]
train_dirs = [
    "AkPnBcht/MUS",
    "AkPnBsdf/MUS",
    "AkPnCGdD/MUS",
    "AkPnStgb/MUS",
    "SptkBGAm/MUS",
    "SptkBGCl/MUS",
    "StbgTGd2/MUS",
]


def generate_training_set(dataset_path: str, dst: str = None):
    """
    Desc:
        Generate the training tensors from the data passed 

    Args:
        dataset_path (string): Directory where unzipped data is located

        dst (optional, string): Path to the directory where the tensors should be created
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} does not exist!")

    if not dst:
        dst = os.path.expanduser("~")
        dst = os.path.join(dst, "storage", "transcription")

    if os.path.isdir(dst):
        dst = os.path.join(dst, "training")
        print(f"Creating training folder {dst}")
        os.mkdir(dst)

    for d in train_dirs:
        # TODO define and point to directories
        path = os.path.join(dataset_path, d)
        path = os.path.join(path, "*.wav")
        wav_files = glob.glob(path)

        # find mid files
        for wav_file in wav_files:
            base_name_root, _ = os.path.splitext(f)
            midi_file = base_name_root + ".mid"

            wav_data, wav_sample_rate = torchaudio.load(wav_file)
            midi_data, midi_sample_rate = torchaudio.load(midi_file)

            torch.save(wav_data, os.path.join(dst, wav_file, ".pt"))
            torch.save(midi_data, os.path.join(dst, midi_file, ".pt"))


def main():

    downloader = Downloader.from_params(preprocess_settings["downloader"])

    dst = preprocess_settings["pre_dst"]

    for d in train_dirs:
        downloader.get_dataset(d, dst)

    for d in test_dirs:
        downloader.get_dataset(d, dst)


if __name__ == "__main__":
    main()
