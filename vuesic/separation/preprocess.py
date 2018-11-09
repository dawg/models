import os
import argparse
import logme
import tqdm
import zipfile
import boto3
import botocore
from boto3.session import Session

import writer

# from separation import writer
# XXX for now we're just going to put these in here but we should export as environment variables
ACCESS_KEY = "xxx"
SECRET_KEY = "xxx"
BUCKET_NAME = "vuesic-musdbd18"  # replace with your bucket name
OBJECT = "musdb18.zip"  # replace with your object key

HOME = os.path.expanduser("~")
DST = os.path.join(HOME, "storage", "separation")

# XXX note that musdb18 only has a training and testing set once installed. We'll have to partition a validation set ourselves.
# Alternatively, we could just find stems from elsewhere to use as a validation set.
class Set:
    TRAIN = "train"
    TEST = "test"


@logme.log
def download_dataset(object, dst, logger=None):

    # Download dst
    path = os.path.join(dst, object)

    if not os.path.isfile(path):

        # XXX Add a loading bar here
        logger.log("Download started")

        # XXX LOAD KEYS FROM ENVIRONMENT HERE (or something)
        session = Session(
            aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
        )

        s3 = session.resource("s3")

        s3.Bucket(BUCKET_NAME).download_file(object, path)

        logger.log("Done!")

    return path


@logme.log
def get_dataset(set: str, path=None, logger=None):

    dst = DST

    if not os.path.exists(dst):
        logger.info("Making {}".format(dst))
        os.makedirs(dst)

    if path == None:
        path = download_dataset(OBJECT, DST)

    with zipfile.ZipFile(path, "r") as z:

        members = []
        for f in z.namelist():
            if f.startswith(set):
                members.append(f)

        z.extractall(path=dst, members=members)

    return


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the musdb18 dataset into a tensorflow record."
    )

    parser.add_argument(
        "--local-path", help="Path to musdb18.zip if it has already been downloaded"
    )
    args = parser.parse_args()

    # extract training and testing sets from archive
    path = args.local_path

    # get_dataset(Set.TRAIN, path)
    # get_dataset(Set.TEST, path)

    writer.write_np(os.path.join(DST, "train"), os.path.join(DST, "np_train"))
    writer.write_np(os.path.join(DST, "test"), os.path.join(DST, "np_test"))


if __name__ == "__main__":
    main()
