from vusic.utils.transcription_settings import constants

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from vusic.utils.transcription_settings import debug
from magenta.protobuf import music_pb2
from google.protobuf.json_format import MessageToJson, Parse


class TranscriptionDataset(Dataset):
    def __init__(self, root_dir: str, transform: callable = None):
        """
        Args:
            root_dir (string): Directory with the training or testing tensors        
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        suffix = ".pth"

        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.filenames = [
            name for name in os.listdir(root_dir) if name.endswith(suffix)
        ]

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

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):

        item_path = os.path.join(self.root_dir, self.filenames[idx])
        chunk_tensor = torch.load(item_path)

        mel = chunk_tensor["mel_spec"]

        ns = music_pb2.NoteSequence()
        Parse(chunk_tensor["ns"], ns)

        velocities = music_pb2.VelocityRange()
        Parse(chunk_tensor["velocities"], velocities)

        sample = {"mel": mel, "ns": ns, "velocities": velocities}

        if self.transform:
            sample = self.transform(sample)

        return sample
