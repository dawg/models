import torch
from torch.utils.data import DataLoader

from vusic.utils.separation_dataset import SeparationDataset
from vusic.utils.separation_settings import debug, hyper_params, training_settings
from vusic.separation.modules import RnnDecoder, RnnEncoder, FnnMasker

# :sunglasses:
# def coolate(batch):
#     return batch

def main():
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
    # fnn_masker = FnnMasker.from_params(training_settings["fnn_masker_params"])
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
        for i, sample in enumerate(dataloader):

            print(f"Sample {i}: {sample['mix']['mg'].shape}")

            mix_mg = sample['mix']['mg'][0, :, :]
            vocal_mg = sample['vocals']['mg'][0, :, :]

            # batch up our song
            for sequence in range(int(mix_mg.shape[0])):

                for batch in range(int()):

                    batch_start = batch * batch_size
                    batch_end = (batch+1) * batch_size

                    mix_mg_batch = mix_mg[batch_start:batch_end, :]
                    vocal_mg_batch = vocal_mg[batch_start:batch_end, :]

                    print(f"batch shape: {mix_mg_batch.shape}")

                    # feed through masker

                    m_enc = rnn_encoder(mix_mg_batch)
                    m_dec = rnn_decoder(m_enc)

                    # feed through twinnet?

                    # regularize twinnet output

                    # feed through denoiser

                    # init optimizer

                    # calculate losses

                    # create objective

                    # back propigation

                    # gradient norm clipping

                    # step through optimizer

                    # record losses
            
    # we are done training! save and record our model



if __name__ == "__main__":
    main()
