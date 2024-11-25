# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Optional, Dict
import torch
from omegaconf import DictConfig

from utils.scheduler import SchedulerSpec
import torch.nn.functional as F
from torch.distributions import Bernoulli


@torch.no_grad()
def stage_free_generation(
    designer,
    logger,
    num_iter: int,
    resample_y_every: int,
    stage_fixedbb_args: Optional[DictConfig] = None,
    resample_y_temp: SchedulerSpec = 1.0,
):
    # greedy-decode - start, i.e., warm-start
    # greedy-decode NOT works, so skip this step!
    # greedy-decode - end
    curr_step = 0
    B, L, K = designer.x_seqs.shape
    while curr_step < num_iter:
        for t in range(L):
            x_seqs = copy.deepcopy(designer.x_seqs)

            e_old = designer.calc_x(x_seqs).gather(
                -1,
                x_seqs.argmax(-1).unsqueeze(-1)).squeeze(-1)
            assert e_old.shape == (
                B,
                L,
            )
            e_old = -(e_old.sum(-1))

            w_o = x_seqs[:, t, :].argmax(-1)
            x_seqs_p = copy.deepcopy(x_seqs)
            x_seqs_p[:, t, :] = F.one_hot(
                torch.full((B,),
                           designer.vocab.mask_idx,
                           device=designer.device), K).float()

            probs = F.softmax(
                designer.calc_mlm(x_seqs_p)['logits'][:, t, :], -1)
            w_n = torch.multinomial(probs, num_samples=1).squeeze(-1)
            x_seqs_p[:, t, :] = F.one_hot(w_n, K).float()

            q_xp_x = probs.gather(-1, w_n.unsqueeze(-1)).squeeze(-1)
            q_x_xp = probs.gather(-1, w_o.unsqueeze(-1)).squeeze(-1)

            e_new = designer.calc_x(x_seqs_p).gather(
                -1,
                x_seqs_p.argmax(-1).unsqueeze_(-1)).squeeze(-1)
            assert e_new.shape == (
                B,
                L,
            )
            e_new = -(e_new.sum(-1))

            #print(-e_new)
            #print(-e_old)
            A_xp_x = ((torch.exp(-e_new - -e_old) * q_x_xp) / q_xp_x).clamp(
                0, 1)
            A_bools = Bernoulli(A_xp_x).sample().bool()  # [B]
            #print(A_bools.shape)
            #exit(0)
            # debug
            if False:
                A_bools = torch.logical_or(
                    A_bools,
                    (x_seqs_p.argmax(-1) == designer.vocab.mask_idx).sum(-1)
                    > 0)

            designer.x_seqs = torch.where(A_bools[:, None, None], x_seqs_p,
                                          x_seqs)  # [B, L, K]
            #exit(0)
            logger.info("e_old: {:.3f}, e_new: {:.3f}".format(e_old.item(),
                                                        e_new.item()))
            for seq in designer.decode(designer.x_seqs):
                logger.info("{: <10}, {: <10}: {}".format(curr_step, t, seq))

        curr_step += 1
        # debug
        #if curr_step % 1 == 0:
