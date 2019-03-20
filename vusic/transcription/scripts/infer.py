import argparse
import os
import sys
import numpy as np
import soundfile
import torch

from pathlib import Path
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm
from ffmpy import FFmpeg
from vusic.transcription.modules.mel import melspectrogram
from vusic.utils.transcription_settings import constants
from vusic.utils.midi_utils import save_midi
from vusic.utils.transcription_utils import (
    extract_notes,
    notes_to_frames,
    summary,
    save_pianoroll,
)


def transcribe(
    audio, audio_path, model, save_path, onset_threshold=0.5, frame_threshold=0.5
):
    mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)

    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    onset_pred.squeeze_(0).relu_()
    offset_pred.squeeze_(0).relu_()
    frame_pred.squeeze_(0).relu_()
    velocity_pred.squeeze_(0).relu_()

    p_est, i_est, v_est = extract_notes(
        onset_pred,
        frame_pred,
        velocity_pred,
        onset_threshold,
        frame_threshold,
    )

    scaling = constants["hop_length"] / constants["sampling_rate"]

    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(constants["min_midi"] + midi) for midi in p_est])

    os.makedirs(save_path, exist_ok=True)
    pred_path = os.path.join(save_path, os.path.basename(audio_path) + ".pred.png")
    save_pianoroll(pred_path, onset_pred, frame_pred)
    midi_path = os.path.join(save_path, os.path.basename(audio_path) + ".pred.mid")
    save_midi(midi_path, p_est, i_est, v_est)


def transcribe_file(
    audio_path, model_file, save_path, onset_threshold=None, frame_threshold=None
):
    audio_extension = Path(audio_path).suffix.lower()

    if audio_extension == ".wav":
        # convert to flac
        wav_path = audio_path
        flac_path = audio_path.replace("wav", ".flac").replace("WAV", "flac")

        ff = FFmpeg(
            inputs={wav_path: "-y -loglevel fatal"},
            outputs={flac_path: "-ac 1 -ar 16000"},
        )

        try:
            ff.run()
            audio_path = flac_path
        except:
            print("ERROR: Failed to convert wav to flac")
            return

    elif audio_extension == ".flac":
        pass
    else:
        print("ERROR: Invalid file format")
        return

    audio, sr = soundfile.read(audio_path, dtype="int16")

    model = torch.load(model_file, map_location="cpu").eval()
    summary(model)

    audio = torch.ShortTensor(audio)
    audio = audio.to('cpu').float().div_(32768.0)
    transcribe(audio, audio_path, model, save_path, onset_threshold, frame_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument("--save-path", default="transcription_output/")
    parser.add_argument("--onset-threshold", default=0.5, type=float)
    parser.add_argument("--frame-threshold", default=0.5, type=float)

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))
