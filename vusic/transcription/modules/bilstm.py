import torch
import torch.nn as nn


__all__ = ["BiLstm"]


class BiLstm(nn.Module):
    def __init__(self, debug):
        """
        Desc:
            Create implementation of the onset BiLSTM, this feeds into a fully connected layer with 88 neurons 

        Args:
            debug - debug mode

            input_size – The number of expected features in the input x

            hidden_size – The number of features in the hidden state h

            num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
            
            bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            
            batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
            
            dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
            
            bidirectional – If True, becomes a bidirectional LSTM. Default: False

        Inputs: 

            input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.

            h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the RNN is bidirectional, num_directions should be 2, else it should be 1.

            c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.

            If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        Outputs:

            output of shape (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.

            For the unpacked case, the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively. Similarly, the directions can be separated in the packed case.

            h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.

            Like output, the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size) and similarly for c_n.

            c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t = seq_len
        """

        super(BiLstm, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.lstm = nn.LSTM(
            512,
            1024,  # Determine hidden layer size
            1,
            bias=False,
            batch_first=False,
            dropout=False,
            bidirectional=True,
        )
        # maps from hidden layer down to # of keys
        self.fnn = nn.Linear(1024, 88)

    def forward(self, x):
        out = self.lstm(x)
        out = self.fnn(out)
        return out
