import os
import argparse
import logme
import tqdm
import zipfile

import musdb

from separation import writer

HOME = os.path.expanduser("~")
DST = os.path.join(HOME, "storage", "separation")

# XXX note that musdb18 only has a training and testing set once installed. We'll have to partition a validation set ourselves.
class Set:
    TRAIN = "train"
    TEST = "test"


# TODO add logger because it is broken for some reason
@logme.log
def get_dataset(set: str, path = None, logger = None):

    dst = DST

    if path == None:        
        # Download to dst


        # Set path equal to download location


    if not os.path.exists(dst):
        # logger.info("Making {}".format(DST))
        os.makedirs(dst)

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

    parser.add_argument("--local-path", help="Path to musdb18.zip")
    args = parser.parse_args()

    # extract training and testing sets from archive
    get_dataset(Set.TRAIN, args.local_path)
    get_dataset(Set.TEST, args.local_path)



if __name__ == "__main__":
    main()
