import math
import time
import os

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import DataLoader

from vusic.utils.separation_dataset import SeparationDataset
from vusic.utils.separation_settings import (
    debug,
    hyper_params,
    training_settings,
    stft_info,
    output_paths,
)

from vusic.separation.modules import RnnDecoder, RnnEncoder, FnnMasker


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
    train_ds = SeparationDataset(training_settings["training_path"])
    dataloader = DataLoader(train_ds, shuffle=True)
    print(f"done! Training set contains {len(train_ds)} samples.", end="\n\n")

    # create nn modules
    print(f"-- Initializing NN modules...", end="")
    rnn_encoder = RnnEncoder.from_params(training_settings["rnn_encoder_params"])
    rnn_decoder = RnnDecoder.from_params(training_settings["rnn_decoder_params"])
    fnn_masker = FnnMasker.from_params(training_settings["fnn_masker_params"])
    print(f"done!", end="\n\n")

    # set up objective functions
    print(f"-- Creating objective functions...", end="")
    l2 = torch.nn.MSELoss()
    print(f"done!", end="\n\n")

    # optimizer
    print(f"-- Creating optimizer...", end="")
    optimizer = O.Adam(
        list(rnn_encoder.parameters())
        + list(rnn_decoder.parameters())
        + list(fnn_masker.parameters()),
        lr=hyper_params["learning_rate"],
    )
    print(f"done!", end="\n\n")

    sequence_length = training_settings["rnn_encoder_params"]["sequence_length"]
    context_length = training_settings["rnn_encoder_params"]["sequence_length"]

    # training in epochs

    # tensors to hold sequence
    mix_mg_sequence = torch.zeros(
        batch_size, sequence_length, stft_info["win_length"], dtype=torch.float
    )
    vocal_mg_sequence = torch.zeros(
        batch_size, sequence_length, stft_info["win_length"], dtype=torch.float
    )

    # create output directory
    if not os.path.exists(output_paths["output_folder"]):
        os.mkdir(output_paths["output_folder"])

    for epoch in range(training_settings["epochs"]):

        epoch_masker_loss = []

        epoch_start = time.time()

        # TODO does the dataloader shuffle every epoch?

        for i, sample in enumerate(dataloader):

            print(f"Sample {i}: {sample['mix']['mg'].size()}")

            mix_mg = sample["mix"]["mg"]
            vocal_mg = sample["vocals"]["mg"]

            # chunk up our song into multiple sequences
            for sequence in range(
                math.floor(mix_mg.shape[1] / (sequence_length * batch_size))
            ):
                sequence_start = sequence * sequence_length
                sequence_end = (sequence + 1) * sequence_length
                for batch in range(sequence_start, sequence_end):

                    batch_start = batch * batch_size
                    batch_end = (batch + 1) * batch_size

                    mix_mg_sequence[:, batch % sequence_length, :] = mix_mg[
                        0, batch_start:batch_end, :
                    ]
                    vocal_mg_sequence[:, batch % sequence_length, :] = vocal_mg[
                        0, batch_start:batch_end, :
                    ]
                # XXX hax
                vocal_mg_sequence_masked = vocal_mg_sequence[:, 10:-10, :]

                # feed through masker
                m_enc = rnn_encoder(mix_mg_sequence)
                m_dec = rnn_decoder(m_enc)
                m_masked = fnn_masker(m_dec, mix_mg_sequence)

                # init optimizer
                optimizer.zero_grad()

                # compute master loss (using KL divergence)
                loss = l2(m_masked, vocal_mg_sequence_masked)
                loss.backward()

                # feed through twinnet?

                # regularize twinnet output

                # feed through denoiser

                # calculate losses

                # create objective

                # back propigation

                # gradient norm clipping
                torch.nn.utils.clip_grad_norm_(
                    list(rnn_encoder.parameters())
                    + list(rnn_decoder.parameters())
                    + list(fnn_masker.parameters()),
                    max_norm=hyper_params["max_grad_norm"],
                    norm_type=2,
                )

                # step through optimizer
                optimizer.step()

                # record losses
                epoch_masker_loss.append(loss.item())

        epoch_end = time.time()

        print(
            f"epoch: {epoch}, masker_loss: {loss}, epoch time: {epoch_end - epoch_start}"
        )
        print(epoch_masker_loss)

    # we are done training! save and record our model state
    torch.save(rnn_encoder, output_paths["rnn_encoder"])
    torch.save(rnn_decoder, output_paths["rnn_decoder"])
    torch.save(fnn_masker, output_paths["fnn_masker"])

    # torch.save(epoch_masker_loss, os.path.join("output", "masker_loss.pth"))


if __name__ == "__main__":
    main()
