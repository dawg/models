import numpy as np
import librosa.core as lr
import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None):
        """
        Desc: 
            create an STFT object

        Args:

        """
        super(STFT, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create an RNN decoder from parameters

        Args:
            param (object): parameters for creating the STFT.
        """

        hop_length = params["hop_length"] if "hop_length" in params else None
        win_length = params["win_length"] if "win_length" in params else None

        return cls(params["n_fft"], hop_length=hop_length, win_length=win_length)

    def forward(self, sample: np.ndarray):
        """
        Desc: 
            Compute the stft of the input sample

        Args:
            sample (numpy.ndarray()): 1D or 2D time domain signal

        Returns:
            Complex values matrix M where:
                np.abs(M[f, t]) is the magnitude of frequency bin f at frame t

                np.angle(M[f, t]) is the phase of frequency bin f at frame t

        Note:
            depending on the length if you take the stft and the istft of a signal, 
            the result will be the original signal minus a couple samples
        """

        return lr.stft(sample, self.n_fft, hop_length=self.hop_length)
