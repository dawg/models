import numpy as np
import librosa.core as lr
import torch
import torch.nn as nn


class ISTFT(nn.Module):
    def __init__(
        self, n_fft, hop_length=None, win_length=None,
    ):
        """
        Desc: 
            create an ISTFT object

        Args:

        """
        super(ISTFT, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create an RNN decoder from parameters

        Args:
            param (object): parameters for creating the ISTFT.
        """

        hop_length = params["hop_length"] if "hop_length" in params else None
        win_length = params["win_length"] if "win_length" in params else None

        return cls(params["n_fft"], hop_length=hop_length, win_length=win_length)
    

    def forward(self, stft: np.ndarray):
        """
        Desc: 
            Compute the stft of the input sample

        Args:
            stft (numpy.ndarray): STFT matrix

        Returns:
            Complex values matrix M where:
                np.abs(M[f, t]) is the magnitude of frequency bin f at frame t

                np.angle(M[f, t]) is the phase of frequency bin f at frame t
        """

        return lr.istft(stft, hop_length=self.hop_length, win_length=self.win_length)