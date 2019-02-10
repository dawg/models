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

hyper_params = {"learning_rate": 1e-4}

# amount of bins we want to preserve
preserved_bins = 744

# context length for RNNs
context_length = 10

output_paths = {
    "rnn_encoder": os.path.join("optput", "states", "rnn_encoder.pt"),
    "rnn_decoder": os.path.join("optput", "states", "rnn_decoder.pt"),
    "fnn_masker": os.path.join("optput", "states", "fnn_masker.pt"),
}

training_settings = {
    "epochs": 2 if debug else 100,
    "training_path": os.path.join(HOME, "storage", "separation", "pt_f_train"),
    "rnn_encoder_params": {
        "debug": debug,
        "input_size": preserved_bins,
        "context_length": context_length,
        "sequence_length": 60,
    },
    "rnn_decoder_params": {"debug": debug, "input_size": preserved_bins * 2},
    "fnn_masker_params": {
        "debug": debug,
        "input_size": preserved_bins * 2,
        "output_size": stft_info["win_length"],
        "context_length": context_length,
    },
    "batch_size": 16,
}
