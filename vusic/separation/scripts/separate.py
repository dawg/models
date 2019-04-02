import argparse
import os
import re
import logme
import time
import librosa as lr
import numpy as np
import torch
import tqdm

from vusic.utils.audio_helper import read_wav, write_wav

# from helpers.data_feeder import data_feeder_testing, data_process_results_testing
from vusic.utils.separation_settings import (
    training_settings,
    debug,
    model_paths,
    stft_info,
)
from vusic.utils import STFT, ISTFT
from vusic.utils.transforms import context_reshape, overlap_sequences
from vusic.separation.modules import RnnDecoder, RnnEncoder, FnnMasker, FnnDenoiser


@logme.log
def separate(source: str, output: str, logger=None):
    """
        Separate audio contained in source
    """

    batch_size = 1
    context_length = training_settings["context_length"]
    sequence_length = training_settings["sequence_length"]
    win_length = stft_info["win_length"]

    device = "cpu"

    logger.info(f"Initializing NN modules")
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

    logger.info("Loading models")
    rnn_encoder.load_state_dict(
        torch.load(model_paths["rnn_encoder"], map_location=device)
    )
    rnn_decoder.load_state_dict(
        torch.load(model_paths["rnn_decoder"], map_location=device)
    )
    fnn_masker.load_state_dict(
        torch.load(model_paths["fnn_masker"], map_location=device)
    )
    fnn_denoiser.load_state_dict(
        torch.load(model_paths["fnn_denoiser"], map_location=device)
    )

    logger.info(f"Reading {source}")
    mix, rate = read_wav(source, mono=True)
    mix = np.transpose(mix).astype(np.float)

    mix_mg, mix_ph = stft(mix)

    # context based reshaping
    mix_mg = overlap_sequences(mix_mg, context_length, sequence_length, batch_size)
    mix_ph = overlap_sequences(mix_ph, context_length, sequence_length, batch_size)

    total_time = 0

    prediction = np.zeros(
        (
            mix_mg.shape[0],
            sequence_length - context_length * 2,
            stft_info["win_length"],
        ),
        dtype=np.float32,
    )

    logger.info("Separating song")
    start_time = time.time()
    for batch in tqdm.tqdm(range(int(mix_mg.shape[0] / batch_size)), unit="Batch"):

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
        prediction[batch_start:batch_end, :, :] = temp_prediction.data.numpy()

    logger.info(f"Saving extracted vocals as {output}")
    prediction.shape = (
        prediction.shape[0] * prediction.shape[1],
        stft_info["win_length"],
    )

    mix_mg, mix_ph = context_reshape(mix_mg, mix_ph, context_length, win_length)

    # take the inverse fourier transform of the vocal prediction
    vocals = istft(prediction, mix_ph)

    write_wav(np.transpose(vocals), rate, 16, output)

    logger.info(f"Total time: {str(time.time() - start_time)}")


def main():
    arg_parser = argparse.ArgumentParser(
        usage="python separate.py the_file.wav",
        description="Script to separate audio!",
    )

    arg_parser.add_argument("source_file", type=str)

    args = arg_parser.parse_args()
    source_file = args.source_file

    print(source_file)
    separate(source_file, "{}_voice.wav".format(source_file))


if __name__ == "__main__":
    main()
