# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import esm


class MaskTokensDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: list[str],
        batch_converter: esm.data.BatchConverter,
        pad_str: str,
        mask_str: str,
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

        self.dataset = dataset
        self.batch_converter = batch_converter
        self.pad_str = pad_str
        self.mask_str = mask_str
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
        item = np.asarray(list(self.dataset[index]), dtype='<U23')
        sz = len(item)

        # decide elements to mask
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * sz + rng.random())

        mask_idc = rng.choice(sz, num_mask, replace=False)
        mask_idc = np.concatenate([mask_idc + 0])
        mask_idc = mask_idc[mask_idc < len(mask)]
        # debug
        #print('mask_idc.shape: {}'.format(mask_idc.shape))
        #print('mask.shape: {}'.format(mask.shape))
        #exit(0)
        try:
            mask[mask_idc] = True
        except:  # something wrong
            print("Assigning mask indexes {} to mask {} failed!".format(
                mask_idc, mask))
            raise

        # the targets for masked LM training)
        tgt_new_item = np.full(len(mask), self.pad_str)
        tgt_new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]

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
                    self.amino_acids,
                    num_rand,
                )
        # debug - lezhang.thu
        _, _, x = self.batch_converter([
            ('protein-masked', ''.join(new_item)),
            ('protein-tgt', ''.join(tgt_new_item)),
        ])
        # debug
        # hacker way!!!
        x[1][0] = x[1][-1] = self.batch_converter.alphabet.padding_idx

        return x[0], x[1]
