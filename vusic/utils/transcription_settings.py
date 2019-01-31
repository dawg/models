import os

__all__ = ["debug", "preprocess_settings", "hyper_params", "training_settings"]

debug = True

HOME = os.path.expanduser("~")

preprocess_settings = {
    "pre_dst": os.path.join(HOME, "storage", "transcription"),
    "downloader": {
        "bucket": "vuesic-musdbd18",
        "dataset": "MAPS.zip",
        "downloading_dst": os.path.join(HOME, "storage", "transcription"),
    },
}

hyper_params = {}

training_settings = {}
