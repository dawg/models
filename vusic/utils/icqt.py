import numpy as np
import librosa.core as lr
import torch
import torch.nn as nn


class CQT(nn.Module):
    def __init__(self, sampling_rate, hop_length=None):
        """
        Desc: 
            create a CQT object

        Args:

        """
        super(CQT, self).__init__()

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create an CQT object from parameters

        Args:
            param (object): parameters for creating the CQT.
        """

        hop_length = params["hop_length"] if "hop_length" in params else None
        sampling_rate = params["sampling_rate"] if "sampling_rate" in params else None

        return cls(hop_length=hop_length, sampling_rate=sampling_rate)

    def forward(self, sample: np.ndarray):
        """
        Desc: 
            Compute the cqt of the input sample

        Args:
            sample (numpy.ndarray()): 1D or 2D time domain signal

        Returns:
            np.ndarray: Constant-Q value for each frequency at each time
        """

        return lr.icqt(
            sample, sampling_rate=self.sampling_rate, hop_length=self.hop_length
        )
