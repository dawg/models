import os
import torch

__all__ = ["debug", "constants", "hyper_params", "training_settings"]

# Constants
debug = True
HOME  = os.path.expanduser("~")

SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 32 // 1000
ONSET_LENGTH = SAMPLE_RATE * 32 // 1000
OFFSET_LENGTH = SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108
N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048
DEFAULT_DEVICE = 'cuda' if not debug and torch.cuda.is_available() else 'cpu'

constants = {
    "sample_rate": SAMPLE_RATE,
    "hop_length": HOP_LENGTH,
    "onset_length": ONSET_LENGTH,
    "offset_length": OFFSET_LENGTH,
    "hops_in_onset": HOPS_IN_ONSET,
    "hops_in_offset": HOPS_IN_OFFSET,
    "min_midi": MIN_MIDI,
    "max_midi": MAX_MIDI,
    "n_mels": N_MELS,
    "mel_fmin": MEL_FMIN,
    "mel_fmax": MEL_FMAX,
    "window_length": WINDOW_LENGTH,
    "default_device": DEFAULT_DEVICE
}

hyper_params = {
    "learning_rate": 1e-4,
}

training_settings = {
    "epochs": 2 if debug else 100,
    "training_path": os.path.join(HOME, "storage", "transcription", "training"),
    "batch_size": 16,
}