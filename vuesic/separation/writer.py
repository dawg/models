import json
import os
from typing import Dict

import logme
import tqdm
from scipy.io import wavfile
import numpy as np
import tensorflow as tf
import musdb
import stempeg


class Stem:
    MIX = (0,)
    DRUMS = (1,)
    BASS = (2,)
    OTHER = (3,)
    VOCALS = (4,)


class Data:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def encode(self, value):
        """
        Encodes the shape as a tf.train.Feature using the correct type. Checks the shape if possible.
        :param value: The value to encode
        :return: The encoded value.
        """
        if self.dtype == tf.string:
            if isinstance(value, str):
                return self._bytes_feature(value.encode())
            elif isinstance(value, list):
                return self._bytes_feature_list(value)
            else:
                return self._bytes_feature(value)

        value = np.asarray(value)
        self.verify_shape(value)

        value = np.array(value).flatten().tolist()
        if self.dtype == tf.float32:
            return self._float32_feature_list(value)
        elif self.dtype == tf.int64:
            return self._int64_feature_list(value)
        elif self.dtype == tf.uint8:
            return self._bytes_feature_list(value)
        else:
            raise NotImplementedError(
                "Encoding for {} not supported.".format(self.dtype)
            )

    def verify_shape(self, value):
        if np.prod(value.shape) != np.prod(self.shape):
            msg = "Shape mismatch for class {}. Given {} expected {}".format(
                self.__class__, value.shape, self.shape
            )
            raise ValueError(msg)

    def decode(self):
        """
        Creates a tf.FixedLenFeature decoder.
        :return: The decoder.
        """
        return tf.FixedLenFeature(self.shape, self.dtype)

    @staticmethod
    def _float32_feature_list(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64_feature_list(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _bytes_feature_list(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class String(Data):
    def __init__(self):
        super().__init__(tf.string, None)


class Floats(Data):
    def __init__(self, shape):
        super().__init__(tf.float32, shape)


class Ints(Data):
    def __init__(self, shape):
        super().__init__(tf.int64, shape)


class Bytes(Data):
    def __init__(self, shape):
        super().__init__(tf.uint8, shape)


def write_record(
    data: dict, writer: tf.python_io.TFRecordWriter, dataset: Dict[str, Data]
):
    features = {}
    for key in dataset:
        features[key] = dataset[key].encode(data[key])

    example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(example.SerializeToString())


@logme.log
def write(src: str, dst: str, logger=None):

    if not os.path.isdir(src):
        raise FileNotFoundError("{} does not exist!".format(src))

    if os.path.isdir(dst):
        raise FileExistsError("{} already exists!".format(dst))

    logger.info("Creating {}".format(dst))
    writer = tf.python_io.TFRecordWriter(os.path.join(dst))
    try:
        logger.info("Reading examples from {}".format(src))
        data = {}
        for fname in tqdm.tqdm(os.listdir(src), unit="Ex"):

            sname = os.path.join(src, fname)

            if not os.path.exists(sname):
                raise FileNotFoundError(f"{sname} not found")

            stem, rate = stempeg.read_stems(sname)

            data["mix"] = np.array(stem[Stem.MIX, :, :].view(dtype=np.int16).tobytes())
            data["vocals"] = np.array(
                stem[Stem.VOCALS, :, :].view(dtype=np.int16).tobytes()
            )

            dataset = {
                "mix": Bytes(data["mix"].shape),
                "vocals": Bytes(data["vocals"].shape),
            }

            write_record(data, writer, dataset)

    except Exception:
        logger.info(f"Removing {dst}")
        if os.path.exists(dst):
            os.remove(dst)
        raise
    finally:
        writer.close()


@logme.log
def write_np(src: str, dst: str, logger=None):

    if not os.path.isdir(src):
        raise FileNotFoundError("{} does not exist!".format(src))

    if not os.path.isdir(dst):
        logger.info("Creating {}".format(dst))
        os.mkdir(dst)

    vocaldst = os.path.join(dst, "vocals")

    if not os.path.exists(vocaldst):
        os.mkdir(vocaldst)

    mixdst = os.path.join(dst, "mix")

    if not os.path.exists(mixdst):
        os.mkdir(mixdst)

    try:
        logger.info("Reading examples from {}".format(src))

        for fname in tqdm.tqdm(os.listdir(src), unit="Ex"):

            sname = os.path.join(src, fname)

            if not os.path.exists(sname):
                raise FileNotFoundError(f"{sname} not found")

            stem, rate = stempeg.read_stems(sname)

            np.save(os.path.join(dst, "vocals", fname), stem[Stem.VOCALS, :, :].astype(np.int16))
            np.save(os.path.join(dst, "mix", fname), stem[Stem.MIX, :, :].astype(np.int16))

    except Exception:
        logger.info(f"Removing {dst}")
        if os.path.exists(dst):
            os.remove(dst)
        raise
