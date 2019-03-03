import torch
import torch.nn as nn


__all__ = ["BiLstm"]


class BiLstm(nn.Module):
    def __init__(self, debug):
        """
        Desc:
            Create implementation of a BiLSTM

        Args:
            debug (bool): debug mode

            input_size – The number of expected features in the input x

            hidden_size – The number of features in the hidden state h

            num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
            
            bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            
            batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
            
            dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
            
            bidirectional – If True, becomes a bidirectional LSTM. Default: False

        """
        super(BiLstm, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.nn.LSTM(input_size, hidden_size, num_layers, bias=False,batch_first=False,dropout=False, bidirectional = True)

    def forward(self, x):
        out = None
        return out
