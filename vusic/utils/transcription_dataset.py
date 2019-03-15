import json
import os
import numpy as np
import soundfile

from abc import abstractmethod
from glob import glob
from vusic.utils.transcription_settings import constants, debug
from vusic.utils.midi_utils import parse_midi
from torch.utils.data import Dataset
from tqdm import tqdm


class TranscriptionDataset(Dataset):
    def __init__(
        self,
        path,
        groups=None,
        sequence_length=None,
        seed=42,
        device=constants["default_device"],
    ):
        self.path = path
        self.groups = groups if groups is not None else self.get_available_groups()
        assert all(group in self.get_available_groups() for group in self.groups)
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)

        self.data = []

        print(
            "Loading %d group%s of %s at %s"
            % (len(groups), "s"[: len(groups) - 1], self.__class__.__name__, path)
        )

        for group in groups:
            for input_files in tqdm(self.files(group), desc="Loading group %s" % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data["path"])

        if self.sequence_length is not None:
            audio_length = len(data["audio"])
            step_begin = (
                self.random.randint(audio_length - self.sequence_length)
                // constants["hop_length"]
            )
            n_steps = self.sequence_length // constants["hop_length"]
            step_end = step_begin + n_steps

            begin = step_begin * constants["hop_length"]
            end = begin + self.sequence_length

            result["audio"] = data["audio"][begin:end].to(self.device)
            result["label"] = data["label"][step_begin:step_end, :].to(self.device)
            result["velocity"] = data["velocity"][step_begin:step_end, :].to(
                self.device
            )
        else:
            result["audio"] = data["audio"].to(self.device)
            result["label"] = data["label"].to(self.device)
            result["velocity"] = data["velocity"].to(self.device).float()

        result["audio"] = result["audio"].float().div_(32768.0)
        result["onset"] = (result["label"] == 3).float()
        result["offset"] = (result["label"] == 1).float()
        result["frame"] = (result["label"] > 1).float()
        result["velocity"] = result["velocity"].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            ramp: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the number of frames after the corresponding onset

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace(".flac", ".pt").replace(".wav", ".pt")
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype="int16")
        assert sr == constants["sampling_rate"]

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = constants["max_midi"] - constants["min_midi"] + 1
        n_steps = (audio_length - 1) // constants["hop_length"] + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(
                round(onset * constants["sampling_rate"] / constants["hop_length"])
            )
            onset_right = min(n_steps, left + constants["hops_in_onset"])
            frame_right = int(
                round(offset * constants["sampling_rate"] / constants["hop_length"])
            )
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + constants["hops_in_offset"])

            f = int(note) - constants["min_midi"]
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data

    def get_available_groups(self):
        return [
            "AkPnBcht",
            "AkPnBsdf",
            "AkPnCGdD",
            "AkPnStgb",
            "SptkBGAm",
            "SptkBGCl",
            "StbgTGd2",
        ]

    def files(self, group):
        flacs = glob(os.path.join(self.path, "flac", "*_%s.flac" % group))
        tsvs = [
            f.replace("/flac/", "/tsv/matched/").replace(".flac", ".tsv") for f in flacs
        ]

        assert all(os.path.isfile(flac) for flac in flacs)
        assert all(os.path.isfile(tsv) for tsv in tsvs)

        return sorted(zip(flacs, tsvs))

    @classmethod
    def from_params(cls, params: object):
        """
        Desc: 
            create a TranscriptionDataset from parameters

        Args:
            param (object): parameters for creating the TranscriptionDataset. Must contain the following
                root_dir (str): root directory of the dataset
                transform (optional, str): transform to be applied to each sample upon retrieval
                training (optional, bool): boolean indicating if this is a training dataset
        """

        transform = params["transform"] if "transform" in params else None

        return cls(params["root_dir"], transform)
