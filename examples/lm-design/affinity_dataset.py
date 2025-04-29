# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import pandas as pd

import esm


class AffinityDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_converter: esm.data.BatchConverter,
        sep_str: str,
        max_len: int,
        seed: int = 1,
    ):
        self.dataset = dataset.drop_duplicates()
        print(
            'within AffinityDataset of affinity_dataset.py, len(self.dataset): {}'
            .format(len(self.dataset)))
        self.batch_converter = batch_converter
        self.sep_str = sep_str
        self.seed = seed
        self.max_len = max_len 
        self.epoch = 0

    def set_epoch(self, epoch):
        # Should be called EVERY epoch!!!
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.__getitem_x__(self.seed, self.epoch, index)

    def __getitem_x__(self, seed: int, epoch: int, index: int):
        seed = int(hash((seed, epoch, index)) % 1e6)
        rng = np.random.default_rng(seed)
        row = self.dataset.iloc[index]
        seq = dict()
        for c in ['A', 'B']:
            seq[c] = ['-'] * self.max_len
            seq[c][:len(row[c])] = list(row[c])

        #item = np.asarray(list(seq['A']) + [self.sep_str] + list(seq['B']))
        item = np.asarray(list(seq['A']) + list(seq['B']))
        _, _, x = self.batch_converter([
            ('protein', ''.join(item)),
        ])
        return x[0], torch.tensor([row['Affinity']], dtype=torch.float32)
