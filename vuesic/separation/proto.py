#%%
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
TEST = os.path.join(HOME, "storage", "separation", "np_test")
TRAIN = os.path.join(HOME, "storage", "separation", "np_train")


#%%
# Sample
fname = os.listdir(os.path.join(TRAIN, "mix"))[0]
x = np.load(os.path.join(src, "mix", fname))
plt.figure(1)
plt.title("Audio Sample: {}".format(fname))
plt.plot(range(1000), np.transpose(x[:, 0:1000, 0]))
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(0, 1000)
plt.show()


#%%
for fname in tqdm.tqdm(os.listdir(os.path.join(src, "mix")), unit="Ex"):
    print(fname)
    x = np.load(os.path.join(src, "mix", fname))
    y = np.load(os.path.join(src, "vocals", fname))

    # do something
    break
