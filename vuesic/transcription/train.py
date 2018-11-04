import os

from magenta.models.nsynth.reader import NSynthDataset

from vuesic.transcription import preprocess


def main():
    train = os.path.join(preprocess.DST, "train.tfrecord")
    valid = os.path.join(preprocess.DST, "valid.tfrecord")

    train = NSynthDataset(train, is_training=True)
    valid = NSynthDataset(valid, is_training=False)

    # do something with the training and validation sets


if __name__ == "__main__":
    main()
