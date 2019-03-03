import glob
import os
import re
import torch
import logme
import zipfile
import torchaudio
import numpy as np
import librosa

from tqdm import tqdm
from vusic.utils.downloader import Downloader
from vusic.utils.transcription_settings import preprocess_settings, log_mel_info
from magenta.music import midi_io, sequences_lib, audio_io
from magenta.protobuf import music_pb2
from google.protobuf.json_format import MessageToJson
from magenta.models.onsets_frames_transcription.split_audio_and_label_data import (
    find_split_points,
)


test_dirs = ["MAPS/ENSTDkCl/MUS", "MAPS/ENSTDkCl/MUS"]
train_dirs = [
    "MAPS/AkPnBcht/MUS",
    "MAPS/AkPnBsdf/MUS",
    "MAPS/AkPnCGdD/MUS",
    "MAPS/AkPnStgb/MUS",
    "MAPS/SptkBGAm/MUS",
    "MAPS/SptkBGCl/MUS",
    "MAPS/StbgTGd2/MUS",
]


def read_wav_file(wave_file: str, target_sampling_rate: int):
    """
    Desc:
        Read the passed in wave file and resample it to the provided sampling
        rate if necessary

    Args:
        wave_file (string): name of the wave file

        target_sampling_rate (string): Target sampling rate for the output wav tensor
    """

    audio_tensor, sr = torchaudio.load(wave_file, out=None)

    if audio_tensor.shape[0] == 2:
        mixer = torchaudio.transforms.DownmixMono(channels_first=True)
        audio_tensor = mixer(audio_tensor)

    audio_np = audio_tensor.numpy()

    if sr != target_sampling_rate:
        audio_np = librosa.resample(audio_np, sr, target_sampling_rate)

    audio_np = librosa.util.normalize(audio_np, norm=np.inf)

    return audio_np.T.squeeze()


def crop_wav_data(wav_data, sample_rate, crop_beginning_seconds, total_length_seconds):
    """Crop WAV sequency.

    Args:
        wav_data: WAV audio data to crop.
        sample_rate: The sample rate at which to read the WAV data.
        crop_beginning_seconds: How many seconds to crop from the beginning of the
            audio.
        total_length_seconds: The desired duration of the audio. After cropping the
            beginning of the audio, any audio longer than this value will be
            deleted.

    Returns:
        A cropped version of the WAV audio.
    """
    samples_to_crop = int(crop_beginning_seconds * sample_rate)
    total_samples = int(total_length_seconds * sample_rate)
    cropped_samples = wav_data[samples_to_crop : (samples_to_crop + total_samples)]
    return cropped_samples


def padarray(A, size):
    t = size - A.shape[0]
    return np.pad(A, (0, t), mode="constant")


def generate_training_set(dataset_path: str, dst: str = None):
    """
    Desc:
        Generate the training tensors from the data passed 

    Args:
        dataset_path (string): Directory where unzipped data is located

        dst (optional, string): Path to the directory where the tensors should be created
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} does not exist!")

    if not dst:
        dst = os.path.expanduser("~")
        dst = os.path.join(dst, "storage", "transcription", "training")

    if not os.path.exists(dst):
        print(f"Creating training folder {dst}")
        os.mkdir(dst)

    for d in tqdm(train_dirs):
        path = os.path.join(dataset_path, d)
        path = os.path.join(path, "*.wav")
        wav_files = glob.glob(path)

        for wav_file in wav_files:
            base_name_root, _ = os.path.splitext(wav_file)
            midi_file = base_name_root + ".mid"

            wav_data = read_wav_file(wav_file, preprocess_settings["sampling_rate"])
            ns = midi_io.midi_file_to_note_sequence(midi_file)

            splits = find_split_points(
                ns,
                wav_data,
                preprocess_settings["sampling_rate"],
                preprocess_settings["min_length"],
                preprocess_settings["max_length"],
            )

            velocities = [note.velocity for note in ns.notes]
            velocity_max = np.max(velocities)
            velocity_min = np.min(velocities)
            velocity_tuple = music_pb2.VelocityRange(min=velocity_min, max=velocity_max)

            base_name = os.path.basename(base_name_root)
            chunk_index = 0

            for start, end in zip(splits[:-1], splits[1:]):
                if end - start < preprocess_settings["min_length"]:
                    continue

                cropped_ns = sequences_lib.extract_subsequence(ns, start, end)
                cropped_wav_data = crop_wav_data(
                    wav_data, preprocess_settings["sampling_rate"], start, end - start
                )

                cropped_wav_data = padarray(
                    cropped_wav_data, preprocess_settings["samples_per_chunk"]
                )

                mel = librosa.feature.melspectrogram(
                    cropped_wav_data,
                    hop_length=log_mel_info["hop_length"],
                    fmin=log_mel_info["fmin"],
                    sr=preprocess_settings["sampling_rate"],
                    n_mels=log_mel_info["n_mels"],
                    htk=log_mel_info["mel_htk"],
                ).astype(np.float32)
                mel = mel.T

                training_sample = {
                    "wav_tensor": torch.from_numpy(cropped_wav_data),
                    "mel_spec": torch.from_numpy(mel),
                    "ns": MessageToJson(cropped_ns),
                    "velocities": MessageToJson(velocity_tuple),
                }

                torch.save(
                    training_sample,
                    os.path.join(dst, base_name + "_" + str(chunk_index) + ".pt"),
                )
                chunk_index += 1

def generate_test_set(dataset_path: str, dst: str = None):
    """
    Desc:
        Generate the testing tensors from the data passed 

    Args:
        dataset_path (string): Directory where unzipped data is located

        dst (optional, string): Path to the directory where the tensors should be created
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} does not exist!")

    if not dst:
        dst = os.path.expanduser("~")
        dst = os.path.join(dst, "storage", "transcription", "testing")

    if not os.path.exists(dst):
        print(f"Creating testing folder {dst}")
        os.mkdir(dst)

    for d in tqdm(test_dirs):
        path = os.path.join(dataset_path, d)
        path = os.path.join(path, "*.wav")
        wav_files = glob.glob(path)

        for wav_file in wav_files:
            base_name_root, _ = os.path.splitext(wav_file)
            midi_file = base_name_root + ".mid"

            wav_data = read_wav_file(wav_file, preprocess_settings["sampling_rate"])
            ns = midi_io.midi_file_to_note_sequence(midi_file)

            splits = find_split_points(
                ns,
                wav_data,
                preprocess_settings["sampling_rate"],
                preprocess_settings["min_length"],
                preprocess_settings["max_length"],
            )

            velocities = [note.velocity for note in ns.notes]
            velocity_max = np.max(velocities)
            velocity_min = np.min(velocities)
            velocity_tuple = music_pb2.VelocityRange(min=velocity_min, max=velocity_max)

            base_name = os.path.basename(base_name_root)
            chunk_index = 0
            for start, end in zip(splits[:-1], splits[1:]):
                if end - start < preprocess_settings["min_length"]:
                    continue

                cropped_ns = sequences_lib.extract_subsequence(ns, start, end)
                cropped_wav_data = crop_wav_data(
                    wav_data, preprocess_settings["sampling_rate"], start, end - start
                )

                cropped_wav_data = padarray(
                    cropped_wav_data, preprocess_settings["samples_per_chunk"]
                )

                mel = librosa.feature.melspectrogram(
                    cropped_wav_data,
                    hop_length=log_mel_info["hop_length"],
                    fmin=log_mel_info["fmin"],
                    sr=preprocess_settings["sampling_rate"],
                    n_mels=log_mel_info["n_mels"],
                    htk=log_mel_info["mel_htk"],
                ).astype(np.float32)
                mel = mel.T

                training_sample = {
                    "wav_tensor": torch.from_numpy(cropped_wav_data),
                    "mel_spec": torch.from_numpy(mel),
                    "ns": MessageToJson(cropped_ns),
                    "velocities": MessageToJson(velocity_tuple),
                }

                torch.save(
                    training_sample,
                    os.path.join(dst, base_name + "_" + str(chunk_index) + ".pt"),
                )
                chunk_index += 1

def main():
    downloader = Downloader.from_params(preprocess_settings["downloader"])
    data_set = preprocess_settings["pre_dst"]
    generate_training_set(data_set)
    generate_test_set(data_set)


if __name__ == "__main__":
    main()
