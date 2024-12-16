# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# core
import logging
import os
import sys
import time

from omegaconf import DictConfig
import hydra
import os
from pathlib import Path
import sys
import time
import logging

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

# make sure script started from the root of the this file
assert Path.cwd(
).name == 'lm-design', 'Please run this script from examples/lm-design/'
sys.path.append('../../')
from esm.data import Alphabet

from utils.scheduler import SchedulerSpec, to_scheduler, set_scheduler_repo
import utils.pdb_loader as pdb_loader
from utils.loss import get_cce_loss
from utils.lm import lm_marginal
from utils.masking import assert_valid_mask
from utils.sampling import (
    set_rng_seeds,)
from utils.constants import COORDS_ANGLE_NAMES, COORDS4D_NAMES
import utils.struct_models as struct_models
from utils.vh_vl_free_generation import stage_free_generation
from utils.fixedbb import stage_fixedbb
from utils.lm import WrapLmEsm

from utils.tensor import (
    assert_shape,)
#from utils import ngram as ngram_utils

logger = logging.getLogger(__name__)  # Hydra configured
os.environ['MKL_THREADING_LAYER'] = 'GNU'


class Designer:
    cutoff_dist = 8
    LOGITS_LARGE = 100
    standard_AA = 'LAGVSERTIDPKQNFYMHWC'

    ##########################################
    # Inits
    ##########################################
    def __init__(self, cfg, target_pdb_path=None, device=None):
        ## Initialize models
        if device:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available() and not cfg.disable_cuda
            device_idx = f":{cfg.cuda_device_idx}" if cfg.get(
                'cuda_device_idx') else ""
            self.device = torch.device(
                f'cuda{device_idx}' if use_cuda else 'cpu')
        SEED_SENTINEL = 1238
        self.seed = cfg.seed + SEED_SENTINEL
        self.cfg = cfg
        self.allowed_AA = ''.join(AA for AA in self.standard_AA
                                  if (('suppress_AA' not in self.cfg) or
                                      (not AA in self.cfg.suppress_AA)))

        self._init_models()
        # debug
        self.vh_max_len = 133
        self.vL_max_len = 122
        self.L = self.vh_max_len + self.vL_max_len

        set_rng_seeds(self.seed)
        self.schedulers = {}  # reset schedulers
        self.resuming_stage = False
        self.init_sequences(cfg.num_seqs)

        torch.backends.cudnn.benchmark = True  # Slightly faster runtime for optimization
        logger.info("Finished Designer init")

    def _init_models(self):
        self.vocab = Alphabet.from_architecture('ESM-1b')
        self.vocab_mask_AA = torch.BoolTensor(
            [t in self.allowed_AA for t in self.vocab.all_toks]).to(self.device)
        self.vocab_mask_AA_idx = torch.nonzero(self.vocab_mask_AA).squeeze()

        self.struct_model, self.pdb_loader_params = struct_models.load(
            self.vocab,)
        self.LM = WrapLmEsm(self.struct_model.lm, self.vocab)

        # 4. Common model settings
        def apply_common_settings(model):
            model.to(self.device)
            model.eval()
            # No grads for models
            for p in model.parameters():
                p.requires_grad = False
            return model

        self.LM = apply_common_settings(self.LM)
        self.struct_model = apply_common_settings(self.struct_model)

        # debug - lezhang.thu
        from esm.pretrained import esm2_t33_650M_UR50D
        self.antibody, _ = esm2_t33_650M_UR50D(use_lora=True)
        lora_missing, lora_unexpected = self.antibody.load_state_dict(
            torch.load(os.path.join('..', '..', '..',
                                    'adapter_5120-VH-VL_aa-llm.pt'),
                       map_location="cpu",
                       weights_only=True),
            strict=False)
        assert all('lora' not in x for x in lora_missing)
        assert len(lora_unexpected) == 0
        self.antibody = WrapLmEsm(self.antibody, self.vocab)
        self.antibody = apply_common_settings(self.antibody)

    def encode(self, seq_raw, onehot=True):
        device = self.device
        if isinstance(seq_raw, list):
            seq_enc = [[self.vocab.get_idx(c) for c in seq] for seq in seq_raw]
        else:
            seq_enc = [self.vocab.get_idx(c) for c in seq_raw]
        seq_enc = torch.LongTensor(seq_enc)
        if onehot:
            seq_enc = F.one_hot(seq_enc, len(self.vocab)).float()
        return seq_enc.to(device)

    def decode(self, seq_enc, onehot=True):
        if onehot:
            seq_enc = seq_enc.argmax(-1)
        # for seq in seq_enc.view(-1, seq_enc.
        assert seq_enc.dim() == 2
        # Must do cpu conversion here!
        # Or else pytorch runtime will do it O(L) times and incur a very
        # large slowdown.
        seq_enc = seq_enc.cpu()
        seqs = [
            ''.join([self.vocab.get_tok(c) for c in _seq]) for _seq in seq_enc
        ]
        return seqs

    def init_sequences(self, num_seqs):
        assert num_seqs == 1, "Only 1 sequence design in parallel supported for now."
        self.B = B = self.num_seqs = num_seqs
        x_seq = ['<mask>']
        self.x_seqs = self.encode([x_seq], onehot=True)
        assert self.x_seqs.shape == (self.B, 1, len(self.vocab))
        # debug - start
        self.vh_prefix = ""
        self.vL_prefix = ""
        # debug - end

    def calc_mlm(
        self,
        x_seqs,
    ):
        return self.antibody(x_seqs)

    def run_from_cfg(self):
        """
        Main run-loop for the Designer. Runs a relevant design procedure from the config.
        """
        logger.info(f'Designing sequence for task: {self.cfg.task}')
        design_cfg = self.cfg.tasks[self.cfg.task]

        mask_token = self.encode([['<mask>']], onehot=True)
        for _ in range(200):
            self.init_sequences(1)
            for idx in range(self.vh_max_len + self.vL_max_len):
                with torch.no_grad():
                    if idx < self.vh_max_len and idx < len(self.vh_prefix):
                        w_n = torch.tensor(
                            [self.vocab.get_idx(self.vh_prefix[idx])],
                            dtype=torch.long,
                            device=self.x_seqs.device)
                    elif idx >= self.vh_max_len and idx - self.vh_max_len < len(
                            self.vL_prefix):
                        w_n = torch.tensor([
                            self.vocab.get_idx(
                                self.vL_prefix[idx - self.vh_max_len])
                        ],
                                           dtype=torch.long,
                                           device=self.x_seqs.device)
                    else:
                        x = self.calc_mlm(self.x_seqs)['logits']
                        probs = F.softmax(x[:, -1, :], -1)
                        #logger.info('idx: {: <5}, probs: {}'.format(idx, probs))
                        w_n = torch.multinomial(probs,
                                                num_samples=1).squeeze(-1)
                    self.x_seqs[:, -1, :] = F.one_hot(w_n,
                                                      len(self.vocab)).float()
                    for seq in self.decode(self.x_seqs):
                        pass
                        #logger.info("{: <10}: {}".format(idx, seq))

                    if idx < self.vh_max_len + self.vL_max_len - 1:
                        self.x_seqs = torch.cat([self.x_seqs, mask_token], 1)
            logger.info(f'Final designed sequences:')
            for seq in self.decode(self.x_seqs):
                logger.info(seq)
            self.output_seq = self.decode(self.x_seqs)[0]


@hydra.main(config_path="conf/", config_name="x-config")
def main(cfg: DictConfig) -> None:
    args_no_spaces = [arg.replace(" ", "") for arg in sys.argv[1:]]
    logger.info(f"Running with args: {' '.join(args_no_spaces)}")

    pdb_fn = cfg.pdb_fn
    logger.info(f'Starting to optimize seq for {pdb_fn}')

    start_time = time.time()
    des = Designer(cfg, pdb_fn)

    des.run_from_cfg()
    logger.info("finished after %s hours", (time.time() - start_time) / 3600)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
