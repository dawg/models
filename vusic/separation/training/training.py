import math
import time
import os

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import DataLoader

import numpy as np
from numpy.lib import stride_tricks

from vusic.utils.separation_dataset import SeparationDataset
from vusic.utils.separation_settings import (
    debug,
    hyper_params,
    training_settings,
    stft_info,
    output_paths,
)
from vusic.utils.objectives import kl, l2, l2_squared, sparse_penalty
from vusic.separation.modules import (
    RnnDecoder,
    RnnEncoder,
    FnnMasker,
    TwinReg,
    AffineTransform,
    FnnDenoiser,
)


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

    def overlap_transform(sample):
        '''
            Make samples overlap by context length frames
        '''

        return sample

    train_ds = SeparationDataset(training_settings["training_path"], transform=overlap_transform)
    dataloader = DataLoader(train_ds, shuffle=True)
    print(f"done! Training set contains {len(train_ds)} samples.", end="\n\n")

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

    # regularization networks
    twin_decoder = TwinReg.from_params(training_settings["twin_decoder_params"]).to(
        device
    )
    twin_masker = FnnMasker.from_params(training_settings["fnn_masker_params"]).to(
        device
    )
    fnn_denoiser = FnnDenoiser.from_params(training_settings["fnn_denoiser_params"]).to(
        device
    )

    # affine transform
    affine_transform = AffineTransform.from_params(
        training_settings["affine_transform_params"]
    ).to(device)

    print(f"done!", end="\n\n")

    # set up objective functions
    print(f"-- Creating objective functions...", end="")

    masker_loss = kl
    twin_loss = kl
    denoiser_loss = kl
    twin_reg = l2
    masker_reg = sparse_penalty
    denoiser_reg = l2_squared

    print(f"done!", end="\n\n")

    # optimizer
    print(f"-- Creating optimizer...", end="")
    optimizer = O.Adam(
        list(rnn_encoder.parameters())
        + list(rnn_decoder.parameters())
        + list(fnn_masker.parameters())
        + list(fnn_denoiser.parameters())
        + list(twin_decoder.parameters())
        + list(twin_masker.parameters())
        + list(affine_transform.parameters()),
        lr=hyper_params["learning_rate"],
    )
    print(f"done!", end="\n\n")

    sequence_length = training_settings["sequence_length"]
    context_length = training_settings["rnn_encoder_params"]["context_length"]

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

        epoch_loss = []

        epoch_start = time.time()

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

                # trim our vocal sequence to match the output of our masker
                vocal_mg_sequence_masked = vocal_mg_sequence[
                    :, context_length:-context_length, :
                ]

                # feed through masker
                m_enc = rnn_encoder(mix_mg_sequence)
                print(f"m_enc: {m_enc.shape}")

                m_dec = rnn_decoder(m_enc)
                print(f"m_dec: {m_dec.shape}")

                m_masked = fnn_masker(m_dec, mix_mg_sequence)
                print(f"m_masked: {m_masked.shape}")

                # feed through twinnet
                m_t_dec = twin_decoder(m_enc)
                print(f"m_t_dec: {m_t_dec.shape}")

                m_t_masked = twin_masker(m_t_dec, mix_mg_sequence)
                print(f"m_t_masked: {m_t_masked.shape}")

                # regulatization
                affine = affine_transform(m_dec)
                print(f"affine: {affine.shape}")

                # denoiser
                denoised = fnn_denoiser(m_masked)
                print(f"denoised: {denoised.shape}")

                # init optimizer
                optimizer.zero_grad()

                # compute loss
                loss_m = masker_loss(m_masked, vocal_mg_sequence_masked)
                loss_twin = twin_loss(m_t_masked, vocal_mg_sequence_masked)
                loss_denoiser = denoiser_loss(denoised, vocal_mg_sequence_masked)

                # compute regularization terms and other penalties
                reg_m = hyper_params["l_reg_m"] * masker_reg(
                    fnn_masker.linear_layer.weight
                )
                reg_twin = hyper_params["l_reg_twin"] * twin_reg(
                    affine, m_t_dec.detach()
                )
                reg_denoiser = hyper_params["l_reg_denoiser"] * denoiser_reg(fnn_denoiser.fnn_dec.weight)

                loss = loss_m + loss_twin + loss_denoiser + reg_m + reg_twin + reg_denoiser


                print(f"loss: {loss}");
                
                loss.backward()

                # gradient norm clipping
                torch.nn.utils.clip_grad_norm_(
                    list(rnn_encoder.parameters())
                    + list(rnn_decoder.parameters())
                    + list(fnn_masker.parameters())
                    + list(fnn_denoiser.parameters())
                    + list(twin_decoder.parameters())
                    + list(twin_masker.parameters())
                    + list(affine_transform.parameters()),
                    max_norm=hyper_params["max_grad_norm"],
                    norm_type=2,
                )

                # step through optimizer
                optimizer.step()

                # record losses
                epoch_loss.append(loss.item())

        torch.save(epoch_loss, output_paths["masker_loss"])
        epoch_end = time.time()

        print(
            f"epoch: {epoch}, masker_loss: {loss}, epoch time: {epoch_end - epoch_start}"
        )
        print(epoch_masker_loss)

    # we are done training! save and record our model state
    torch.save(rnn_encoder, output_paths["rnn_encoder"])
    torch.save(rnn_decoder, output_paths["rnn_decoder"])
    torch.save(fnn_masker, output_paths["fnn_masker"])

    torch.save(epoch_loss, output_paths["masker_loss"])
    # torch.save(epoch_twin_loss, masker_loss["twin_loss"])


if __name__ == "__main__":
    main()
