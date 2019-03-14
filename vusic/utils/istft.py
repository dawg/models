import numpy as np
import librosa.core as lr
import torch
import torch.nn as nn


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None):
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

    def forward(self, magnitude: np.ndarray, phase: np.ndarray):
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
        
        rs = _gl_alg(window_size, hop, (window_size - 1) * 2)

        hw_1 = int(np.floor((window_size + 1) / 2))
        hw_2 = int(np.floor(window_size / 2))

        # Acquire the number of STFT frames
        nb_frames = magnitude_spect.shape[0]

        # Initialise output array with zeros
        time_domain_signal = np.zeros(nb_frames * hop + hw_1 + hw_2)

        # Initialise loop pointer
        pin = 0

        # Main Synthesis Loop
        for index in range(nb_frames):
            # Inverse Discrete Fourier Transform
            y_buf = _i_dft(magnitude_spect[index, :], phase[index, :], window_size)

            # Overlap and Add
            time_domain_signal[pin:pin + window_size] += y_buf * rs

            # Advance pointer
            pin += hop

        # Delete the extra zeros that the analysis had placed
        time_domain_signal = np.delete(time_domain_signal, range(3 * hop))
        time_domain_signal = np.delete(
            time_domain_signal,
            range(time_domain_signal.size - (3 * hop + 1),
                time_domain_signal.size)
        )

        return time_domain_signal
        

        return lr.istft(stft, hop_length=self.hop_length, win_length=self.win_length)
