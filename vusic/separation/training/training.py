import torch
from torch.utils.data import DataLoader

from vusic.utils.separation_dataset import SeparationDataset
from vusic.utils.separation_settings import debug, hyper_params, training_settings
from vusic.separation.modules import RnnDecoder, RnnEncoder, FnnMasker


def main():
    device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

    print(f"\n-- Starting training. Debug mode: {debug}")
    print(f"-- Using: {device}", end="\n\n")

    # init dataset
    print(f"-- Loading training data...", end="")
    train_ds = SeparationDataset(training_settings["training_path"])
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    print(f"done! Training set contains {len(train_ds)} samples.", end="\n\n")

    # create nn modules
    print(f"-- Initializing NN modules...", end="")
    rnn_encoder = RnnEncoder.from_params(training_settings["rnn_encoder_params"])
    rnn_decoder = RnnDecoder.from_params(training_settings["rnn_decoder_params"])
    fnn_masker = FnnMasker.from_params(training_settings["fnn_masker_params"])
    print(f"done!", end="\n\n")

    # set up objective functions
    print(f"-- Creating objective functions...", end="")
    print(f"done!", end="\n\n")

    # obj1 = l2

    # optimizer
    print(f"-- Creating optimizer...", end="")
    print(f"done!", end="\n\n")

    # training in epochs
    for epoch in range(training_settings["epochs"]):
        for i, data in enumerate(dataloader):
            print(f"Sample {i}: {sample['mix'][0].shape}")


if __name__ == "__main__":
    main()
