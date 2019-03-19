"""Usage script.
"""

import argparse
import os

import time

import librosa as lr

import numpy as np
import torch

from vusic.utils.audio_helper import read_wav, write_wav

# from helpers.data_feeder import data_feeder_testing, data_process_results_testing
from vusic.utils.separation_settings import training_settings, debug, model_paths, stft_info
from vusic.utils import STFT, ISTFT
from vusic.utils.transforms import context_reshape, overlap_sequences
from vusic.separation.modules import (
    RnnDecoder,
    RnnEncoder,
    FnnMasker,
    FnnDenoiser,
)
def separate(source: str, output: str):
    """
    Separate audio
    """
    
    batch_size = 1
    context_length = training_settings["context_length"]
    sequence_length = training_settings["sequence_length"]

    device = 'cuda' if not debug and torch.cuda.is_available() else 'cpu'

    print(f"Using: {device}")

    print(f"-- Initializing NN modules...", end="")
    # masker
    rnn_encoder = RnnEncoder.from_params(training_settings["rnn_encoder_params"]).to(
        device
    )
    rnn_decoder = RnnDecoder.from_params(training_settings["rnn_decoder_params"]).to(
        device
    )
    fnn_masker = FnnMasker.from_params(training_settings["fnn_masker_params"]).to(
        device
    )

    # denoiser
    fnn_denoiser = FnnDenoiser.from_params(training_settings["fnn_denoiser_params"]).to(
        device
    )

    # stft and istft
    stft = STFT.from_params(stft_info)
    istft = ISTFT.from_params(stft_info)

    print(f"done!", end="\n\n")

    print("-- Loading models...")
    rnn_encoder.load_state_dict(torch.load(model_paths['rnn_encoder']))
    rnn_decoder.load_state_dict(torch.load(model_paths['rnn_decoder']))
    fnn_masker.load_state_dict(torch.load(model_paths['fnn_masker']))
    fnn_denoiser.load_state_dict(torch.load(model_paths['fnn_denoiser']))
    print(f"done!", end="\n\n")

    mix, rate = read_wav(source)
    mix = np.transpose(mix).astype(np.float)
    print(f"len: {mix.shape}")
    mix = lr.to_mono(mix);

    mix_mg, mix_ph = stft(mix)

    print(f"stft: {mix_mg.shape}")

    # context based reshaping
    mix_mg = overlap_sequences(mix_mg, context_length, sequence_length, batch_size)
    mix_ph = overlap_sequences(mix_ph, context_length, sequence_length, batch_size)

    total_time = 0

    print(f"after overlap: {mix_mg.shape}")


    win_length = stft_info["win_length"]

    prediction = np.zeros(
    (
        mix_mg.shape[0],
        sequence_length - context_length * 2,
        stft_info['win_length']
    ),
    dtype=np.float32)

    print("-- Separating song...")
    start_time = time.time()
    for batch in range(int(mix_mg.shape[0]/batch_size)):

        batch_start = batch * batch_size
        batch_end = (batch + 1) * batch_size

        mix_mg_sequence = torch.from_numpy(mix_mg[batch_start:batch_end, :, :])

        # feed through masker
        temp_prediction = rnn_encoder(mix_mg_sequence)
        temp_prediction = rnn_decoder(temp_prediction)
        temp_prediction = fnn_masker(temp_prediction, mix_mg_sequence)

        # denoiser
        temp_prediction = fnn_denoiser(temp_prediction)

        # append the batch prediction to our vocal preciction
        prediction[batch_start:batch_end, :, :] = temp_prediction.data.cpu().numpy()
    
    print("-- Saving extracted vocals...")
    prediction.shape = (prediction.shape[0] * prediction.shape[1], stft_info['win_length'])

    mix_mg, mix_ph = context_reshape(mix_mg, mix_ph, context_length, win_length)

    # take the inverse fourier transform of the vocal prediction
    vocals = istft(prediction, mix_ph)

    write_wav(np.transpose(vocals), rate, 16 , output)
    print(vocals.shape)

    print(f"Total time: {time.time()-start_time}")

def main():
    arg_parser = argparse.ArgumentParser(
        usage='python separate.py [-f the_file.wav]|',
        description='Script to separate audio!'
    )
    
    arg_parser.add_argument(
        '--file', '-f', action='store', dest='source_file', default='',
        help='Source to separate.'
    )

    args = arg_parser.parse_args()
    source_file = args.source_file

    print(f"separating: {source_file}");

    separate(source_file, '{}_voice.wav'.format(source_file))


if __name__ == '__main__':
    main()