import os

__all__ = [
    "debug",
    "preprocess_settings",
    "stft_info",
    "hyper_params",
    "training_settings",
]

debug = True

HOME = os.path.expanduser("~")

preprocess_settings = {
    "pre_dst": os.path.join(HOME, "storage", "separation"),
    "downloader": {
        "bucket": "vuesic-musdbd18",
        "dataset": "musdb18.zip",
        "downloading_dst": os.path.join(HOME, "storage", "separation"),
    },
}

stft_info = {"n_fft": 4096, "win_length": 2049, "hop_length": 384}

hyper_params = {
    "learning_rate": 1e-5,
    "max_grad_norm": 0.05,
    "l_reg_m": 1e-2,
    "l_reg_twin": 0.5,
    "l_reg_denoiser": 1e-4,
}

# amount of bins we want to preserve
preserved_bins = 744

# context length for RNNs
context_length = 10
output_folder = "output"

output_paths = {
    "output_folder": output_folder,
    "rnn_encoder": os.path.join(output_folder, "rnn_encoder.pth"),
    "rnn_decoder": os.path.join(output_folder, "rnn_decoder.pth"),
    "fnn_masker": os.path.join(output_folder, "fnn_masker.pth"),
    "masker_loss": os.path.join(output_folder, "masker_loss.pth"),
    "twin_loss": os.path.join(output_folder, "twin_loss.pth"),
}

training_settings = {
    "epochs": 2 if debug else 100,
    "training_path": os.path.join(HOME, "storage", "separation", "pt_f_train"),
    "context_length": context_length,
    "sequence_length": 60,
    "rnn_encoder_params": {
        "debug": debug,
        "input_size": preserved_bins,
        "context_length": context_length,
    },
    "rnn_decoder_params": {"debug": debug, "input_size": preserved_bins * 2},
    "twin_decoder_params": {"debug": debug, "input_size": preserved_bins * 2},
    "fnn_masker_params": {
        "debug": debug,
        "input_size": preserved_bins * 2,
        "output_size": stft_info["win_length"],
        "context_length": context_length,
    },
    "fnn_denoiser_params": {
        "input_size": stft_info["win_length"],
        "debug": debug,

    },
    "affine_transform_params": {"debug": debug, "input_size": preserved_bins * 2},
    "batch_size": 16,
}


# def _make_overlap_sequences(mixture, voice, bg, l_size, o_lap, b_size):
#     """Makes the overlap sequences to be used for time-frequency transformation.

#     :param mixture: The mixture signal
#     :type mixture: numpy.core.multiarray.ndarray
#     :param voice: The voice signal
#     :type voice: numpy.core.multiarray.ndarray
#     :param bg: The background signal
#     :type bg: numpy.core.multiarray.ndarray
#     :param l_size: The context length in frames
#     :type l_size: int
#     :param o_lap: The overlap in samples
#     :type o_lap: int
#     :param b_size: The batch size
#     :type b_size: int
#     :return: The overlapping sequences
#     :rtype: numpy.core.multiarray.ndarray
#     """
#     trim_frame = mixture.shape[0] % (l_size - o_lap)
#     trim_frame -= (l_size - o_lap)
#     trim_frame = np.abs(trim_frame)

#     if trim_frame != 0:
#         mixture = np.pad(mixture, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
#         voice = np.pad(voice, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
#         bg = np.pad(bg, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))

#     mixture = stride_tricks.as_strided(
#         mixture,
#         shape=(int(mixture.shape[0] / (l_size - o_lap)), l_size, mixture.shape[1]),
#         strides=(mixture.strides[0] * (l_size - o_lap), mixture.strides[0], mixture.strides[1])
#     )
#     mixture = mixture[:-1, :, :]

#     voice = stride_tricks.as_strided(
#         voice,
#         shape=(int(voice.shape[0] / (l_size - o_lap)), l_size, voice.shape[1]),
#         strides=(voice.strides[0] * (l_size - o_lap), voice.strides[0], voice.strides[1])
#     )
#     voice = voice[:-1, :, :]

#     bg = stride_tricks.as_strided(
#         bg,
#         shape=(int(bg.shape[0] / (l_size - o_lap)), l_size, bg.shape[1]),
#         strides=(bg.strides[0] * (l_size - o_lap), bg.strides[0], bg.strides[1])
#     )
#     bg = bg[:-1, :, :]

#     b_trim_frame = (mixture.shape[0] % b_size)
#     if b_trim_frame != 0:
#         mixture = mixture[:-b_trim_frame, :, :]
#         voice = voice[:-b_trim_frame, :, :]
#         bg = bg[:-b_trim_frame, :, :]

#     return mixture, voice, bg