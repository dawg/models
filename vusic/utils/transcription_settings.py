import os

__all__ = ["debug", "preprocess_settings", "hyper_params", "training_settings"]

debug = True

HOME = os.path.expanduser("~")

preprocess_settings = {
    "pre_dst": os.path.join(HOME, "storage", "transcription"),
    "downloader": {
        "bucket": "vuesic-musdbd18",
        "dataset": "MAPS",
        "downloading_dst": os.path.join(HOME, "storage", "transcription"),
    },
    "sampling_rate": 16000,
    "min_length": 5,
    "max_length": 20.0,
    "samples_per_chunk": 320000,
}

log_mel_info = {"n_mels": 226, "hop_length": 512, "mel_htk": False, "fmin": 30}

hyper_params = {}

training_settings = {
    "epochs": 2 if debug else 100,
    "training_path": os.path.join(HOME, "storage", "transcription", "training"),
    "testing_path": os.path.join(HOME, "storage", "transcription", "testing"),
    "batch_size": 16,
}
