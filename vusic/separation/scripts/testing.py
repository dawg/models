import math
import time
import os

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import DataLoader
from mir_eval import separation as evaluation

import numpy as np

from vusic.utils.separation_dataset import SeparationDataset
from vusic.utils.separation_settings import (
    debug,
    hyper_params,
    training_settings,
    testing_settings,
    model_paths,
    stft_info,
    output_paths,
)

from vusic.utils.transforms import (
    overlap_transform_testing,
    context_reshape,
    overlap_sequences,
)
from vusic.utils.objectives import kl, l2
from vusic.utils import ISTFT
from vusic.utils.transforms import overlap_transform
from vusic.separation.modules import RnnDecoder, RnnEncoder, FnnMasker, FnnDenoiser


def main():
    # set seed for consistency
    torch.manual_seed(5)

    device = "cpu"

    device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

    # batch_size = training_settings["batch_size"]
    batch_size = 1
    context_length = training_settings["context_length"]
    win_length = stft_info["win_length"]

    print(f"\n-- Starting training. Debug mode: {debug}")
    print(f"-- Using: {device}", end="\n\n")

    # init dataset
    print(f"-- Loading training data...", end="")
    test_ds = SeparationDataset(testing_settings["testing_path"])
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

    istft = ISTFT.from_params(stft_info)
    print(f"done!", end="\n\n")

    print("-- Loading models...")
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
    print(f"done!", end="\n\n")

    sequence_length = training_settings["sequence_length"]
    context_length = training_settings["rnn_encoder_params"]["context_length"]

    print("-- Starting Testing")

    test_start = time.time()

    sdr = []
    sar = []
    sir = []
    isr = []

    for i, sample in enumerate(dataloader):
        mix_mg = sample["mix"]["mg"]
        mix_ph = sample["mix"]["ph"]
        voice_mg = sample["vocals"]["mg"]
        voice_ph = sample["vocals"]["ph"]

        print(f"{mix_mg.shape}")

        voice = istft(voice_mg[0].data.numpy(), voice_ph[0].data.numpy())
        mixture = istft(mix_mg[0].data.numpy(), mix_ph[0].data.numpy())

        print(f"{voice.shape}")

        mix_mg_o = overlap_sequences(
            mix_mg[0], context_length, sequence_length, batch_size
        )

        print(f"{mix_mg_o.shape}")

        prediction = np.zeros(
            (
                mix_mg_o.shape[0],
                sequence_length - context_length * 2,
                stft_info["win_length"],
            ),
            dtype=np.float32,
        )

        for batch in range(int(mix_mg_o.shape[0] / batch_size)):
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size

            mix_mg_sequence = torch.from_numpy(mix_mg_o[batch_start:batch_end, :, :])

            # feed through masker
            temp_prediction = rnn_encoder(mix_mg_sequence)
            temp_prediction = rnn_decoder(temp_prediction)
            temp_prediction = fnn_masker(temp_prediction, mix_mg_sequence)

            # denoiser
            temp_prediction = fnn_denoiser(temp_prediction)
            prediction[batch_start:batch_end, :, :] = temp_prediction.data.numpy()

        print(f"prediction: {prediction.shape}")

        prediction.shape = (
            prediction.shape[0] * prediction.shape[1],
            stft_info["win_length"],
        )

        mix_mg_r, mix_ph_r = context_reshape(
            mix_mg.data.numpy(), mix_ph, context_length, win_length
        )

        voice_hat = istft(prediction, mix_ph_r)
        minlen = min(len(voice_hat), len(voice))

        # take the inverse fourier transform of the vocal prediction
        background = np.add(-voice, mixture)
        background_hat = np.add(-voice_hat[:minlen], mixture[:minlen])

        print(
            f"prediction: {prediction.shape}, voice_hat: {voice_hat.shape}, voice: {voice.shape}"
        )

        (
            temp_sdr,
            temp_isr,
            temp_sir,
            temp_sar,
            temp_perm,
        ) = evaluation.bss_eval_images_framewise(
            [voice[:minlen], background[:minlen]],
            [voice_hat[:minlen], background_hat[:minlen]],
        )
        sdr_s = (
            np.median([i for i in temp_sdr[0] if not np.isnan(i) and not np.isinf(i)]),
        )
        isr_s = (
            np.median([i for i in temp_isr[0] if not np.isnan(i) and not np.isinf(i)]),
        )
        sir_s = (
            np.median([i for i in temp_sir[0] if not np.isnan(i) and not np.isinf(i)]),
        )
        sar_s = (
            np.median([i for i in temp_sar[0] if not np.isnan(i) and not np.isinf(i)]),
        )
        print(f"SDR: {sdr_s}, ISR: {isr_s}, SIR: {sir_s}, SAR: {sar_s}")

        sdr.append(sdr_s)
        isr.append(isr_s)
        sir.append(sir_s)
        sar.append(sar_s)

        print(
            f"Mean SDR: {np.mean(sdr)}, Mean ISR: {np.mean(isr)}, Mean SIR: {np.mean(sir)}, Mean SAR: {np.mean(sar)}"
        )
    test_end = time.time()


if __name__ == "__main__":
    main()
