import torch

import numpy as np
from numpy.lib import stride_tricks

from vusic.utils.separation_settings import training_settings

_epsilon = np.finfo(np.float32).tiny


def context_reshape(mix, voice, context_length, win_length):
    """
        Reshape based on context
    """
    mix = np.ascontiguousarray(
        mix[:, context_length:-context_length, :], dtype=np.float32
    )
    mix.shape = (mix.shape[0] * mix.shape[1], win_length)
    voice = np.ascontiguousarray(
        voice[:, context_length:-context_length, :], dtype=np.float32
    )
    voice.shape = (voice.shape[0] * voice.shape[1], win_length)

    return mix, voice


def overlap_transform(sample):
    """
        Make samples overlap by context length frames. return the transformed sample
    """

    # Fixme
    sequence_length = training_settings["sequence_length"]
    context_length = training_settings["context_length"]
    batch_size = training_settings["batch_size"]

    sample["mix"]["mg"] = overlap_sequences(
        sample["mix"]["mg"], context_length, sequence_length, batch_size
    )
    sample["vocals"]["mg"] = overlap_sequences(
        sample["vocals"]["mg"], context_length, sequence_length, batch_size
    )

    return sample


def overlap_transform_testing(sample):
    """
        Make samples overlap by context length frames. return the transformed sample
    """
    sequence_length = training_settings["sequence_length"]
    context_length = training_settings["context_length"]
    batch_size = training_settings["batch_size"]

    sample["mix"]["mg"] = overlap_sequences(
        sample["mix"]["mg"], context_length, sequence_length, batch_size
    )

    sample["vocals"]["mg"] = overlap_sequences(
        sample["vocals"]["mg"], context_length, sequence_length, batch_size
    )
    return sample


def overlap_sequences(spectrum, context_length, sequence_length, batch_size):
    """
        Make spectrum overlap by context length frames. return the transformed spectrum
    """

    overlap = context_length * 2

    trim_frame = spectrum.shape[0] % (sequence_length - overlap)
    trim_frame -= sequence_length - overlap
    trim_frame = np.abs(trim_frame)

    if trim_frame != 0:
        spectrum = np.pad(
            spectrum, ((0, trim_frame), (0, 0)), "constant", constant_values=(0, 0)
        )

    spectrum = stride_tricks.as_strided(
        spectrum,
        shape=(
            int(spectrum.shape[0] / (sequence_length - overlap)),
            sequence_length,
            spectrum.shape[1],
        ),
        strides=(
            spectrum.strides[0] * (sequence_length - overlap),
            spectrum.strides[0],
            spectrum.strides[1],
        ),
    )
    spectrum = spectrum[:-1, :, :]

    b_trim_frame = spectrum.shape[0] % batch_size
    if b_trim_frame != 0:
        spectrum = spectrum[:-b_trim_frame, :, :]

    return spectrum


def ideal_masking(mixture_in, magn_spectr_target, magn_spectr_residual):
    """
    Computation of Ideal Amplitude Ratio Mask. As appears in :\
    H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,\
    `Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,`\
    in ICASSP 2015, Brisbane, April, 2015.
    """
    mask = np.divide(
        magn_spectr_target, (_epsilon + magn_spectr_target + magn_spectr_residual)
    )
    return np.multiply(mask, mixture_in)
