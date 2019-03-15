import torch
import torch.nn as nn


__all__ = ["BiLstm"]


class BiLstm(nn.Module):
    def __init__(self, debug, in_features, recr_features):
        """
        Desc:
            Create implementation of the onset BiLSTM, this feeds into a fully connected layer with 88 neurons 

        Args:
            debug - debug mode

            in_features ():

            out_ features (): 

        """
        inference_chunk_length = 512
        super(BiLstm, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"
        self.rnn = nn.LSTM(
            in_features, recr_features, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(
                batch_size,
                sequence_length,
                num_directions * hidden_size,
                device=x.device,
            )

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

        return output
