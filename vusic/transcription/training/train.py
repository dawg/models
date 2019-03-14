import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from vusic.utils.transcription_dataset import TranscriptionDataset
from vusic.transcription.modules.model import Model
from vusic.utils.transcription_settings import (
    debug,
    hyper_params,
    training_settings,
    output_paths,
)

def main():
    # set seed for consistency
    torch.manual_seed(5)
    
    device = "cuda" if not debug and torch.cuda.is_available() else "cpu"
    batch_size = training_settings["batch_size"]

    print(f"\n-- Starting training. Debug mode: {debug}")
    print(f"-- Using: {device}", end="\n\n")

    # init dataset
    print(f"-- Loading training data...", end="")
    train_ds = TranscriptionDataset(training_settings["training_path"])
    dataloader = DataLoader(train_ds, shuffle=True)
    print(f"done! Training set contains {len(train_ds)} samples.", end="\n\n")

    print(f"-- Initializing The model...", end="")
    model = Model(True).to(device)
    print(model)
    print(f"done!", end="\n\n")
    
    # optimizer
    print(f"-- Creating optimizer...", end="")
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyper_params["learning_rate"],
    )
    print(f"done!", end="\n\n")

    for epoch in range(training_settings["epochs"]):

        epoch_loss = []

        epoch_start = time.time()

        for i, sample in enumerate(dataloader):

            print(f"Sample {i}: {sample['mix']['mg'].size()}")

            mel = sample["mel"].to(device)
            ns = sample["ns"].to(device)
            velocities = sample["velocities"]

            # Init Optimizers
            optimizer.zero_grad()
            mel = mel.requires_grad_() #set requires_grad to True for training

            onset_output = model(mel)
            frame_output = model(mel, onset_output)

        epoch_end = time.time()

        print(
            f"epoch: {epoch}, loss: {loss}, epoch time: {epoch_end - epoch_start}"
        )


if __name__ == 'main':
   main()