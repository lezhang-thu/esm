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
        VH_VL_SIZE = 300
        half = VH_VL_SIZE // 2

        t = vh_vl
        vh_vl = dict()
        vh_vl['VH_aa'] = [self.unk_str] * half
        vh_vl['VH_aa'][:len(t['VH_aa'])] = list(t['VH_aa'])

        vh_vl['VL_aa'] = [self.unk_str] * half
        vh_vl['VL_aa'][:len(t['VL_aa'])] = list(t['VL_aa'])
        # debug - end

        item = np.asarray(list(vh_vl['VH_aa']) + list(vh_vl['VL_aa']),
                          dtype='<U23')
        sz = len(item)

        # decide elements to mask
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * sz + rng.random())

        mask_idc = rng.choice(sz, num_mask, replace=False)
        # debug
        #print('mask_idc.shape: {}'.format(mask_idc.shape))
        #print('mask.shape: {}'.format(mask.shape))
        #exit(0)
        mask[mask_idc] = True

        # the targets for masked LM training
        tgt_new_item = np.full(len(mask), self.pad_str)
        tgt_new_item[mask] = item[mask]
        #x_mask = np.copy(mask)

        # decide unmasking and random replacement
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (rng.random(sz) < rand_or_unmask_prob)
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = rng.random(sz) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        new_item = np.copy(item)
        new_item[mask] = self.mask_str
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                new_item[rand_mask] = rng.choice(
                    self.amino_acids + [self.unk_str],
                    num_rand,
                )
        # critical - start
        #vh_vl_item = np.full(VH_VL_SIZE, self.pad_str, dtype='<U23')
        #vh_vl_item[:len(vh_vl['VH_aa'])] = new_item[:len(vh_vl['VH_aa'])]
        #vh_vl_item[half:half +
        #           len(vh_vl['VL_aa'])] = new_item[len(vh_vl['VH_aa']):]

        #vh_vl_tgt_item = np.full(VH_VL_SIZE, self.pad_str)
        #vh_vl_tgt_item[:len(vh_vl['VH_aa'])] = tgt_new_item[:len(vh_vl['VH_aa']
        #                                                        )]
        #vh_vl_tgt_item[half:half +
        #               len(vh_vl['VL_aa'])] = tgt_new_item[len(vh_vl['VH_aa']):]
        vh_vl_item = new_item
        vh_vl_tgt_item = tgt_new_item
        # critical - end

        # debug - lezhang.thu
        _, _, x = self.batch_converter([
            ('protein-masked', ''.join(vh_vl_item)),
            ('protein-tgt', ''.join(vh_vl_tgt_item)),
        ])
        # debug
        # hacker way!!!
        x[1][0] = x[1][-1] = self.batch_converter.alphabet.padding_idx
        ## <cls> + seq. + <eos> ALL ignored!
        #x_mask = np.concatenate([[False], x_mask, [False]])
        #x[1][~x_mask] = -1

        return x[0], x[1]
