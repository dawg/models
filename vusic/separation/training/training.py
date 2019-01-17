import torch

from vusic.utils.separation_dataset import SeparationDataset
from vusic.utils.separation_settings import debug, hyper_params, training_settings
from vusic.separation.modules import RnnDecoder, RnnEncoder


def main():
    device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

    print(f"\n-- Starting training. Debug mode: {debug}")
    print(f"-- Using: {device}", end="\n\n")

    # init dataset
    print(f"-- Loading training data...", end="")
    train_ds = SeparationDataset(training_settings["training_path"])
    print(f"done! Training set contains {len(train_ds)} samples.", end="\n\n")

    # create nn modules
    # rnn_encoder = RnnEncoder.from_params(training_settings["rnn_encoder_params"])
    rnn_decoder = RnnDecoder.from_params(training_settings["rnn_decoder_params"])

    # set up objective functions
    # obj1 = l2

    # optimizer

    # training in epochs
    for epoch in range(training_settings["epochs"]):
        print("hello training")


if __name__ == "__main__":
    main()
