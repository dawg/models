import numpy as np
from scipy import fftpack, signal
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
            Compute the istft from the magnitude and phase spectrograms

        Args:
            stft (numpy.ndarray): STFT matrix

        Returns:
            Complex values matrix M where:
                np.abs(M[f, t]) is the magnitude of frequency bin f at frame t

                np.angle(M[f, t]) is the phase of frequency bin f at frame t
        """

        rs = _gl_alg(self.win_length, self.hop_length, (self.win_length - 1) * 2)

        hw_1 = int(np.floor((self.win_length + 1) / 2))
        hw_2 = int(np.floor(self.win_length / 2))

        # Acquire the number of STFT frames
        nb_frames = magnitude.shape[0]

        # Initialise output array with zeros
        time_domain_signal = np.zeros(nb_frames * self.hop_length + hw_1 + hw_2)

        # Initialise loop pointer
        pin = 0

        # Main Synthesis Loop
        for index in range(nb_frames):
            # Inverse Discrete Fourier Transform
            y_buf = _i_dft(magnitude[index, :], phase[index, :], self.win_length)

            # Overlap and Add
            time_domain_signal[pin : pin + self.win_length] += y_buf * rs

            # Advance pointer
            pin += self.hop_length

        # Delete the extra zeros that the analysis had placed
        time_domain_signal = np.delete(time_domain_signal, range(3 * self.hop_length))
        time_domain_signal = np.delete(
            time_domain_signal,
            range(
                time_domain_signal.size - (3 * self.hop_length + 1),
                time_domain_signal.size,
            ),
        )

        return time_domain_signal


def _i_dft(magnitude_spect, phase, window_size):
    """
        Inverse Discrete Fourier Transform 
    """
    # Get FFT Size
    fft_size = magnitude_spect.size
    fft_points = (fft_size - 1) * 2

    # Half of window size parameters
    hw_1 = int(np.floor((window_size + 1) / 2))
    hw_2 = int(np.floor(window_size / 2))

    # Initialise output spectrum with zeros
    tmp_spect = np.zeros(fft_points, dtype=complex)
    # Initialise output array with zeros
    time_domain_signal = np.zeros(window_size)

    # Compute complex spectrum(both sides) in two steps
    tmp_spect[0:fft_size] = magnitude_spect * np.exp(1j * phase)
    tmp_spect[fft_size:] = magnitude_spect[-2:0:-1] * np.exp(-1j * phase[-2:0:-1])

    # Perform the iDFT
    fft_buf = np.real(fftpack.ifft(tmp_spect))

    # Roll-back the zero-phase windowing technique
    time_domain_signal[:hw_2] = fft_buf[-hw_2:]
    time_domain_signal[hw_2:] = fft_buf[:hw_1]

    return time_domain_signal


def _gl_alg(window_size, hop, fft_size=4096):
    """
        Compute ideal synthesis window

        According to: Daniel W. Griffin and Jae S. Lim, `Signal estimation\
        from modified short-time Fourier transform,` IEEE Transactions on\
        Acoustics, Speech and Signal Processing, vol. 32, no. 2, pp. 236-243,\
        Apr 1984.
    """
    syn_w = signal.hamming(window_size) / np.sqrt(fft_size)
    syn_w_prod = syn_w ** 2.0
    syn_w_prod.shape = (window_size, 1)
    redundancy = int(window_size / hop)
    env = np.zeros((window_size, 1))

    for k in range(-redundancy, redundancy + 1):
        env_ind = hop * k
        win_ind = np.arange(1, window_size + 1)
        env_ind += win_ind

        valid = np.where((env_ind > 0) & (env_ind <= window_size))
        env_ind = env_ind[valid] - 1
        win_ind = win_ind[valid] - 1
        env[env_ind] += syn_w_prod[win_ind]

    syn_w = syn_w / env[:, 0]

    return syn_w
