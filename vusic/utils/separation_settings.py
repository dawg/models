import os

__all__ = ["debug", "hyper_params", "training_settings"]

debug = True

hyper_params = {}

HOME = os.path.expanduser("~")

training_settings = {
    "epochs": 2 if debug else 100,
    "training_path": os.path.join(HOME, "storage", "separation", "pt_f_train"),
    "rnn_decoder_params": {"debug": debug, "in_dim": 12},
    # "rnn_encoder_params": {"debug": debug, "in_dim": 12},
}
