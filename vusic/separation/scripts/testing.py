import math
import time
import os

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import DataLoader

import numpy as np

from vusic.utils.separation_dataset import SeparationDataset
from vusic.utils.separation_settings import (
    debug,
    hyper_params,
    training_settings,
    stft_info,
    output_paths,
)
from vusic.utils.transforms import overlap_transform
from vusic.utils.objectives import kl, l2
from vusic.separation.modules import RnnDecoder, RnnEncoder, FnnMasker, FnnDenoiser


def main():
    # set seed for consistency
    torch.manual_seed(5)

    device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

    batch_size = training_settings["batch_size"]
    context_length = training_settings["rnn_encoder_params"]["context_length"]

    print(f"\n-- Starting training. Debug mode: {debug}")
    print(f"-- Using: {device}", end="\n\n")

    # init dataset
    print(f"-- Loading training data...", end="")
    test_ds = SeparationDataset(
        training_settings["training_path"], transform=overlap_transform
    )
    dataloader = DataLoader(test_ds, shuffle=True)
    print(f"done! Testing set contains {len(test_ds)} samples.", end="\n\n")

    # create nn modules
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
    print(f"done!", end="\n\n")

    print("-- Loading models...")
    rnn_encoder.load_state_dict(torch.load(output_paths["rnn_encoder"])).to(device)
    rnn_decoder.load_state_dict(torch.load(output_paths["rnn_decoder"])).to(device)
    fnn_masker.load_state_dict(torch.load(output_paths["fnn_masker"])).to(device)
    fnn_denoiser.load_state_dict(torch.load(output_paths["fnn_denoiser"])).to(device)
    print(f"done!", end="\n\n")

    sequence_length = training_settings["sequence_length"]
    context_length = training_settings["rnn_encoder_params"]["context_length"]

    print("-- Starting Testing")

    test_start = time.time()

    for i, sample in enumerate(dataloader):

        print(f"-- Sample {i}: {sample['fname']}, {sample['mix']['mg'].size()}")

        mix_mg = sample["mix"]["mg"]

        print(f"batches in song: {int(mix_mg.shape[1]/batch_size)}")

        for batch in range(int(mix_mg.shape[1] / batch_size)):

            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size

            mix_mg_sequence = mix_mg[0, batch_start:batch_end, :, :]

            # feed through masker
            temp_prediction = rnn_encoder(mix_mg_sequence)
            temp_prediction = rnn_decoder(temp_prediction)
            temp_prediction = fnn_masker(temp_prediction, mix_mg_sequence)

            # denoiser
            temp_prediction = fnn_denoiser(temp_prediction)

    test_end = time.time()


if __name__ == "__main__":
    main()
