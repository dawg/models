import glob
import os
import re
import torch
import logme
import tqdm
import zipfile
import boto3
import torch
import threading


class Downloader:
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
    def download_dataset(self, bucket: str, filename: str, dst: str, logger=None):
        """
       Desc: 
       Download a dataset to dst from the s3 bucket described by bucket

       Args:
          bucket (string): name of the aws bucket that has the file
          
          filename (string): filename to be retrieved from our bucket

          dst (string): download directory

       logger (object, optional): logger object (taken care of by decorator)
       """
        path = os.path.join(dst, filename)

        if not os.path.isfile(path):

            logger.info("Download started")

            bucket = boto3.resource("s3").Bucket(bucket)

            pbar = self.CallbackProgressBar(
                bucket.Object(filename).get()["ContentLength"], unit="bytes"
            )

            try:
                bucket.download_file(filename, path, Callback=pbar)
            except Exception:
                logger.info(f"Failed to download {filename}")
                raise

        return path

    @logme.log
    def get_dataset(self, set: str, path: str = None, logger: object = None):
        """
       Desc:
          Retrieve the dataset. If it isn't available, download it

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
            path = self.download_dataset(OBJECT, DST)

        with zipfile.ZipFile(path, "r") as z:

            logger.info(f"Extracting files from {path}")
            for fname in tqdm.tqdm(z.namelist(), unit="Ex"):
                if fname.startswith(set) and not os.path.exists(
                    os.path.join(dst, fname)
                ):
                    z.extractall(path=dst, members=[fname])

        return
