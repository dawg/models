import torch

import numpy as np
from numpy.lib import stride_tricks

from vusic.utils.separation_settings import training_settings

sequence_length = training_settings['sequence_length']
context_length = training_settings['context_length']
batch_size = training_settings['batch_size']


def overlap_transform(sample):
    '''
        Make samples overlap by context length frames. return the transformed sample
    '''

    trim_frame = sample['mix']['mg'].shape[0] % (sequence_length - context_length)
    trim_frame -= (sequence_length - context_length)
    trim_frame = np.abs(trim_frame)

    if trim_frame != 0:
        sample['mix']['mg'] = np.pad(sample['mix']['mg'], ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        sample['vocals']['mg'] = np.pad(sample['vocals']['mg'], ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))

    sample['mix']['mg'] = stride_tricks.as_strided(
        sample['mix']['mg'],
        shape=(int(sample['mix']['mg'].shape[0] / (sequence_length - context_length)), sequence_length, sample['mix']['mg'].shape[1]),
        strides=(sample['mix']['mg'].strides[0] * (sequence_length - context_length), sample['mix']['mg'].strides[0], sample['mix']['mg'].strides[1])
    )
    sample['mix']['mg'] = sample['mix']['mg'][:-1, :, :]
    
    sample['vocals']['mg'] = stride_tricks.as_strided(
        sample['vocals']['mg'],
        shape=(int(sample['vocals']['mg'].shape[0] / (sequence_length - context_length)), sequence_length, sample['vocals']['mg'].shape[1]),
        strides=(sample['vocals']['mg'].strides[0] * (sequence_length - context_length), sample['vocals']['mg'].strides[0], sample['vocals']['mg'].strides[1])
    )
    sample['vocals']['mg'] = sample['vocals']['mg'][:-1, :, :]

    b_trim_frame = (sample['mix']['mg'].shape[0] % batch_size)
    if b_trim_frame != 0:
        sample['mix']['mg'] = sample['mix']['mg'][:-b_trim_frame, :, :]
        sample['vocals']['mg'] = sample['vocals']['mg'][:-b_trim_frame, :, :]

    sample['mix']['mg'] = torch.clamp(torch.from_numpy(sample['mix']['mg']), 0, 1)
    sample['vocals']['mg'] = torch.clamp(torch.from_numpy(sample['vocals']['mg']), 0, 1)


    return sample