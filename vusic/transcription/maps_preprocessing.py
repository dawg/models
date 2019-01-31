import glob
import os
import re
import torch
import logme
import tqdm
import zipfile
import boto3

from magenta.music import audio_io
from magenta.music import midi_io

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


@logme.log
def download_maps_dataset(key: str, dst: str, logger=None):
    """
   Desc: 
      Download the MUSDB18 dataset to dst from the s3 bucket described by the following 
      BUCKET_NAME (string)
      OBJECT (string)

   Args:
      key (string): filename to be retrieved from our bucket

      dst (string): download directory

      logger (object, optional): logger object (taken care of by decorator)
   """
