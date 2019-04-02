import argparse
import os
import sys
import numpy as np
import soundfile
import torch

from pathlib import Path
from mir_eval.util import midi_to_hz
from ffmpy import FFmpeg
from vusic.transcription.modules.mel import melspectrogram
from vusic.utils.transcription_settings import constants, inference_settings
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
        onset_pred, frame_pred, velocity_pred, onset_threshold, frame_threshold
    )

    scaling = constants["hop_length"] / constants["sampling_rate"]

    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(constants["min_midi"] + midi) for midi in p_est])

    os.makedirs(save_path, exist_ok=True)
    midi_path = os.path.join(save_path, os.path.splitext(os.path.basename(audio_path))[0] + ".mid")
    save_midi(midi_path, p_est, i_est, v_est)


def transcribe_file(
    audio_path,
    model_file=inference_settings["trained_model_dir"],
    save_path=None,
    onset_threshold=0.5,
    frame_threshold=0.5,
):
    audio_extension = Path(audio_path).suffix.lower()

    if audio_extension == ".wav":
        # convert to flac
        wav_path = audio_path
        flac_path = audio_path.replace(".wav", ".flac").replace(".WAV", ".flac")

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

    audio = torch.ShortTensor(audio)
    audio = audio.to("cpu").float().div_(32768.0)

    if save_path is None:
        save_path = os.path.dirname(audio_path)

    transcribe(audio, audio_path, model, save_path, onset_threshold, frame_threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
