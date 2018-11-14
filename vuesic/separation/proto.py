#%%
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from SeparationDataset import SeparationDataset

# Interactive plotting
plt.ion()

HOME = os.path.expanduser("~")
TEST = os.path.join(HOME, "storage", "separation", "np_test")
TRAIN = os.path.join(HOME, "storage", "separation", "np_train")
ds = SeparationDataset(TRAIN)

#%%
nsamples = 1000
idx = 0
sample = ds[idx]
x = sample["mix"]
y = sample["vocals"]

f, subplot = plt.subplots(2, sharex=True, sharey=True)

subplot[0].set_title("Audio Sample: {}".format(ds.filenames[idx]))
subplot[0].set_xlim(0, nsamples)
subplot[0].plot(range(nsamples), np.transpose(x[:, 0:nsamples, 0]))
subplot[0].set_ylabel("Mix")
subplot[1].plot(range(nsamples), np.transpose(y[:, 0:nsamples, 0]))
subplot[1].set_ylabel("Vocals")
