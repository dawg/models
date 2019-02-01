import glob
import os
import re
import torch
import logme
import tqdm
import zipfile
import boto3
from botocore.client import Config
import torch
import threading


class Downloader:
    def __init__(self, dataset: str, bucket: str, zip_archive: str = None):
        self.dataset = dataset
        self.bucket = bucket
        self.zip_archive = zip_archive

    @classmethod
    def from_params(cls, params: object):
        """
        Desc: 
            create a downloader from parameters

        Args:
            param (object): parameters for creating the RNN. Must contain the following
                bucket_name (str): AWS bucket name

                dataset_name (str): AWS object name

                zip_archive (str): path to the zip archive if it is already downloaded
        """

        zip_archive = params["zip_archive"] if "zip_archive" in params else None

        return cls(params["dataset"], params["bucket"], zip_archive=zip_archive)

    class CallbackProgressBar(object):
        def __init__(self, total: int, unit: str = None):
            """
            Desc:
                Class that acts as a tqdm progress bar you can pass as a callback

            Args:
                total (int): Total number of iterations

            unit (string, optional): Unit with respect to each iteration
            """
            self.pbar = tqdm.tqdm(total=total, unit=unit)
            self.lock = threading.Lock()

        def __call__(self, update: int):
            """
          Desc:
             Updates the progress bar

          Args:
             update (int): Iterations completed since last update
          """
            with self.lock:
                self.pbar.update(update)

    @logme.log
    def download_dataset(self, dst: str, logger=None):
        """
        Desc: 
        Download a dataset to dst from the s3 bucket described by bucket

        Args:

        logger (object, optional): logger object (taken care of by decorator)
        """
        path = os.path.join(dst, self.dataset)

        if not os.path.isfile(path):

            logger.info("Download started")

            bucket = boto3.resource("s3").Bucket(self.bucket)

            pbar = self.CallbackProgressBar(
                bucket.Object(self.dataset).get()["ContentLength"], unit="bytes"
            )

            try:
                bucket.download_file(self.dataset, path, Callback=pbar)
            except Exception:
                logger.info(f"Failed to download {self.dataset}")
                raise

        return path

    @logme.log
    def get_dataset(self, directory: str, dst: str, logger: object = None):
        """
       Desc:
          Retrieve the dataset. If it isn't available, download it

          Args:
             directory (string): directory to be retrieved from our bucket

             dst (string): destination to extrac the dataset to

             logger (object, optional): logger
       """

        if not os.path.exists(dst):
            logger.info(f"Making {dst}")
            os.makedirs(dst)


        path = self.download_dataset(dst)

        with zipfile.ZipFile(path, "r") as z:

            logger.info(f"Extracting files from {path}")
            for fname in tqdm.tqdm(z.namelist(), unit="Ex"):
                if fname.startswith(directory) and not os.path.exists(os.path.join(dst, fname)):
                    print("***" + fname + "***")
                    logger.info(f"file in {path}")
                    z.extractall(path=dst, members=[fname])

        return
