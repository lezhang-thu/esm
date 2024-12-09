# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import pandas as pd

import esm


class MaskTokensDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_converter: esm.data.BatchConverter,
        pad_str: str,
        mask_str: str,
        unk_str: str,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset.drop_duplicates()
        print(
            'within MaskTokensDataset of vh_vl_mask_tokens_dataset.py, len(self.dataset): {}'
            .format(len(self.dataset)))
        self.batch_converter = batch_converter
        self.pad_str = pad_str
        self.mask_str = mask_str
        self.unk_str = unk_str
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        # debug - amino acids - twenty
        self.amino_acids = [
            'A',
            'R',
            'N',
            'D',
            'C',
            'E',
            'Q',
            'G',
            'H',
            'I',
            'L',
            'K',
            'M',
            'F',
            'P',
            'S',
            'T',
            'W',
            'Y',
            'V',
        ]
        self.vh_max_len = 133
        self.vL_max_len = 122
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
        vh_vl = self.dataset.iloc[index]

        # debug - start
        t = vh_vl
        vh_vl = dict()
        vh_vl['VH_aa'] = ['-'] * self.vh_max_len
        vh_vl['VH_aa'][:len(t['VH_aa'])] = list(t['VH_aa'])

        vh_vl['VL_aa'] = ['-'] * self.vL_max_len
        vh_vl['VL_aa'][:len(t['VL_aa'])] = list(t['VL_aa'])
        # debug - end

        item = np.asarray(list(vh_vl['VH_aa']) + list(vh_vl['VL_aa']),
                          dtype='<U23')
        prefix_idx = rng.choice(self.vh_max_len + self.vL_max_len - 1)
        item = item[:prefix_idx + 1]
        tgt_item = np.full(prefix_idx + 1, self.pad_str)
        tgt_item[-1] = item[-1]
        item[-1] = self.mask_str

        # debug - lezhang.thu
        _, _, x = self.batch_converter([
            ('protein-masked', ''.join(item)),
            ('protein-tgt', ''.join(tgt_item)),
        ])
        # debug
        # hacker way!!!
        x[1][0] = x[1][-1] = self.batch_converter.alphabet.padding_idx

        return x[0], x[1]
