import torch
import torch.nn as nn
from vusic.transcription.modules.kelz_cnn import KelzCnn
from vusic.transcription.modules.bilstm import BiLstm


__all__ = ["OnsetFrameModel"]


class OnsetFrameModel(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super(OnsetFrameModel, self).__init__()

        fc_size = model_complexity * 16
        lstm_units = model_complexity * 8

        self.onset_stack = nn.Sequential(
            KelzCnn(input_features, fc_size),
            BiLstm(fc_size, lstm_units),
            nn.Linear(lstm_units * 2, output_features),
            nn.Sigmoid(),
        )
        self.offset_stack = nn.Sequential(
            KelzCnn(input_features, fc_size),
            BiLstm(fc_size, lstm_units),
            nn.Linear(lstm_units * 2, output_features),
            nn.Sigmoid(),
        )
        self.frame_stack = nn.Sequential(
            KelzCnn(input_features, fc_size),
            nn.Linear(fc_size, output_features),
            nn.Sigmoid(),
        )
        self.combined_stack = nn.Sequential(
            BiLstm(output_features * 3, lstm_units),
            nn.Linear(lstm_units * 2, output_features),
            nn.Sigmoid(),
        )
        self.velocity_stack = nn.Sequential(
            KelzCnn(input_features, fc_size), nn.Linear(fc_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat(
            [onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1
        )
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, mel):
        audio_label = batch["audio"]
        onset_label = batch["onset"]
        offset_label = batch["offset"]
        frame_label = batch["frame"]
        velocity_label = batch["velocity"]

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        predictions = {
            "onset": onset_pred.reshape(*onset_label.shape),
            "offset": offset_pred.reshape(*offset_label.shape),
            "frame": frame_pred.reshape(*frame_label.shape),
            "velocity": velocity_pred.reshape(*velocity_label.shape),
        }

        losses = {
            "loss/onset": F.binary_cross_entropy(predictions["onset"], onset_label),
            "loss/offset": F.binary_cross_entropy(predictions["offset"], offset_label),
            "loss/frame": F.binary_cross_entropy(predictions["frame"], frame_label),
            "loss/velocity": self.velocity_loss(
                predictions["velocity"], velocity_label, onset_label
            ),
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (
                onset_label * (velocity_label - velocity_pred) ** 2
            ).sum() / denominator
