import os
import torch

from datetime import datetime

__all__ = ["debug", "constants", "training_settings", "inference_settings"]

# Constants
debug = False

HOME = os.path.expanduser("~")

SAMPLING_RATE = 16000
HOP_LENGTH = SAMPLING_RATE * 32 // 1000
ONSET_LENGTH = SAMPLING_RATE * 32 // 1000
OFFSET_LENGTH = SAMPLING_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108
N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLING_RATE // 2
WINDOW_LENGTH = 2048
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

constants = {
    "sampling_rate": SAMPLING_RATE,
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
    "default_device": DEFAULT_DEVICE,
}

training_settings = {
    "epochs": 2 if debug else 100,
    "training_path": os.path.join(HOME, "storage", "transcription", "data", "MAPS"),
    "batch_size": 8,
    "iterations": 100000,
    "resume_iteration": 45000,
    "checkpoint_interval": 5000,
    "sequence_length": 327680,
    "model_complexity": 48,
    "learning_rate": 0.0006,
    "learning_rate_decay_steps": 10000,
    "learning_rate_decay_rate": 0.98,
    "clip_gradient_norm": 3,
    "bilstm_inference_chunk_length": 512,
    "model_dir": "/runs/transcription-190320-205138",
}

inference_settings = {"trained_model_dir": os.path.join(HOME, "Desktop/model-45000.pt")}
