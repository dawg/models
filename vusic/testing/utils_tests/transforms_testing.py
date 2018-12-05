from vusic.utils.separation_dataset import SeparationDataset
import os
import PyQt5
import matplotlib.pyplot as plt

if __name__ == "__main__":
    HOME = os.path.expanduser("~")
    TEST = os.path.join(HOME, "storage", "separation", "pt_test")
    TRAIN = os.path.join(HOME, "storage", "separation", "pt_train")
    train_ds = SeparationDataset(TRAIN)
    test_ds = SeparationDataset(TEST)
    print(f"Training set contains {len(train_ds)} samples.")
    print(f"Testing set contains {len(test_ds)} samples.")

    stft = STFT()
    istft = ISTFT()

    nsamples = 10000
    window_width = 512

    idx = 0

    # grab a random sample
    sample = train_ds[idx]

    # convert to "mono"
    thing = sample["mix"][:, 0:nsamples, 0]

    magnitude, phase, ac = stft.forward(Variable(thing))
    print(f"{magnitude.shape}, {phase.shape}, {ac.shape}")

    reconstruction = istft.forward(magnitude, phase, ac)
    print(f"{reconstruction.shape}")

    f, subplot = plt.subplots(4, sharex=True)

    subplot[0].set_title("Audio Sample: {}".format(train_ds.filenames[idx]))
    subplot[0].set_xlim(0, nsamples)
    subplot[0].plot(range(nsamples), torch.t(thing).numpy()[0:nsamples])
    subplot[0].set_ylabel("Mix")
    subplot[1].plot(
        range(window_width), np.transpose(magnitude.detach().numpy()[0, :, :, 0])
    )
    subplot[1].set_ylabel("STFT Mix (Magnitude)")
    subplot[2].plot(
        range(window_width), np.transpose(phase.detach().numpy()[0, :, :, 0])
    )
    subplot[2].set_ylabel("STFT Mix (Phase)")
    subplot[3].plot(
        range(reconstruction.shape[1]),
        torch.t(reconstruction.detach()).numpy()[0:nsamples],
    )
    subplot[3].set_ylabel("Reconstructed Mix (Phase)")

    plt.show()
