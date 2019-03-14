import numpy as np
from scipy import fftpack, signal
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

    def forward(self, x: np.ndarray):
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

        windowing_func = signal.hamming(self.win_length, True)

        window_size = windowing_func.size

        x = np.append(np.zeros(3 * self.hop_length), x)
        x = np.append(x, np.zeros(3 * self.hop_length))

        p_in = 0
        p_end = x.size - window_size
        indx = 0

        if np.sum(windowing_func) != 0.0:
            windowing_func = windowing_func / np.sqrt(self.n_fft)

        xm_x = np.zeros(
            (int(len(x) / self.hop_length), int(self.n_fft / 2) + 1), dtype=np.float32
        )
        xp_x = np.zeros(
            (int(len(x) / self.hop_length), int(self.n_fft / 2) + 1), dtype=np.float32
        )

        while p_in <= p_end:
            x_seg = x[p_in : p_in + window_size]

            mc_x, pc_x = _dft(x_seg, windowing_func, self.n_fft)

            xm_x[indx, :] = mc_x
            xp_x[indx, :] = pc_x

            p_in += self.hop_length
            indx += 1

        return xm_x, xp_x


def _dft(x, windowing_func, fft_size):
    """
    Discrete Fourier Transformation(Analysis) of a given real input signal.
    """
    half_n = int(fft_size / 2) + 1

    hw_1 = int(np.floor((windowing_func.size + 1) / 2))
    hw_2 = int(np.floor(windowing_func.size / 2))

    win_x = x * windowing_func

    fft_buffer = np.zeros(fft_size)
    fft_buffer[:hw_1] = win_x[hw_2:]
    fft_buffer[-hw_2:] = win_x[:hw_2]

    x = fftpack.fft(fft_buffer)

    magn_x = np.abs(x[:half_n])
    phase_x = np.angle(x[:half_n])

    return magn_x, phase_x
