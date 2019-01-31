import glob
import os
import re
import torch
import logme
import tqdm
import zipfile
import boto3

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


def generate_training_set(exclude_ids):
    """
    Desc:
        Generate the training tensors

    Args:
        exclude_ids (string): Desc
    """
    # train_file_pairs = []
    # for d in train_dirs:
    #     #TODO define and point to directories
    #     path = os.path.join(input_dir, directory)
    #     path = os.path.join(path, '*.wav')
    #     wav_files = glob.glob(path)
    pass    




def main():
    downloader = Downloader.from_params(preprocess_settings["downloader"])
    
    dst = preprocess_settings["pre_dst"]

    for d in train_dirs:
        downloader.get_dataset(d, dst)
    
    for d in test_dirs:
        downloader.get_dataset(d,dst)
    

if __name__ == "__main__":
    main()
