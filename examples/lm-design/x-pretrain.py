# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# core
import logging
import os
import sys
import time
import random

from omegaconf import DictConfig
import hydra
import os
from pathlib import Path
import sys
import time
import logging
from typing import Dict

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
import utils.struct_models as struct_models

from utils.tensor import (
    assert_shape,)

logger = logging.getLogger(__name__)  # Hydra configured
os.environ['MKL_THREADING_LAYER'] = 'GNU'


def get_adapter_params(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to an adapter.
    Assumes that any adapter class has defined the
    :func:`~torchtune.modules.peft.AdapterModule.adapter_params` method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        Dict[str, nn.Parameter]: the subset of model's state dict containing
        only adapter parameters.

    """
    adapter_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "adapter_params") and callable(v.adapter_params):
            current_adapter_params = v.adapter_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_adapter_params:
                    full_key = f"{k}.{n}" if k else n
                    adapter_params.update({full_key: p})
                    current_adapter_params.remove(n)
            assert (current_adapter_params == []
                   ), f"Adapter params {current_adapter_params} not converted"
    return adapter_params


class Designer:
    cutoff_dist = 8
    LOGITS_LARGE = 100
    standard_AA = 'LAGVSERTIDPKQNFYMHWC'

    ##########################################
    # Inits
    ##########################################
    def __init__(self, cfg, device=None):
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

        set_rng_seeds(self.seed)
        self.schedulers = {}  # reset schedulers

        torch.backends.cudnn.benchmark = True  # Slightly faster runtime for optimization
        logger.info("Finished Designer init")

    def _init_models(self):
        self.vocab = Alphabet.from_architecture('ESM-1b')
        from esm.pretrained import esm2_t33_650M_UR50D
        self.LM, _ = esm2_t33_650M_UR50D(use_lora=True)

        # debug - start
        lora_missing, lora_unexpected = self.LM.load_state_dict(
            torch.load(os.path.join('..', '..', '..',
                                    'adapter_512-VH-VL_aa.pt'),
                       map_location="cpu",
                       weights_only=True),
            strict=False)
        assert all('lora' not in x for x in lora_missing)
        assert len(lora_unexpected) == 0
        # debug - end

        # 4. Common model settings
        def apply_common_settings(model):
            model.to(self.device)
            model.eval()
            # No grads for models
            for p in model.parameters():
                p.requires_grad = False
            return model

        self.LM = apply_common_settings(self.LM)
        # debug
        self.adapter_params = get_adapter_params(self.LM)
        #print(self.adapter_params)

    def run_from_cfg(self):
        import pandas as pd
        logger.info(os.getcwd())

        # Read the CSV file
        df = pd.read_csv(
            '/home/ubuntu/lezhang.thu/biology-research/covid/esm/examples/lm-design/antibody9_16.csv'
        )

        # Get the column 'VH_aa' and convert it to a set of unique values
        from pad_mask_tokens_dataset import MaskTokensDataset
        ds = MaskTokensDataset(
            #dataset=list(set(df['VH_aa'])),
            #dataset=list(set(df['VL_aa'])),
            dataset=df[['VH_aa', 'VL_aa']],
            batch_converter=self.vocab.get_batch_converter(),
            pad_str="<pad>",
            mask_str="<mask>",
            unk_str="<unk>",
            seed=self.seed,
            mask_prob=0.15,
            leave_unmasked_prob=0.1,
            random_token_prob=0.1,
        )
        num_epochs = 512
        batch_size = 1
        grad_acc = 16
        train_dataloader = torch.utils.data.DataLoader(ds,
                                                       batch_size=batch_size,
                                                       shuffle=True)
        # optimizer
        self.LM.train()
        for name, param in self.LM.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        lora_params = [
            param for name, param in self.LM.named_parameters()
            if 'lora' in name
        ]
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=1e-4,
            weight_decay=0.01,
        )

        # loss
        xent = torch.nn.CrossEntropyLoss(
            ignore_index=self.vocab.get_idx("<pad>"))
            #ignore_index=-1)

        for e in range(num_epochs):
            # SET_EPOCH!!!
            ds.set_epoch(e)
            for idx, (masked, tgt) in enumerate(train_dataloader):
                masked = masked.cuda()
                tgt = tgt.cuda()

                logits = self.LM(masked)['logits'].transpose(1, 2)
                # debug
                #print('logits.shape: {}'.format(logits.shape))
                x = xent(logits, tgt)
                if random.uniform(0, 1) < 1e-2:
                    logger.info(
                        'epoch: {: <5}, idx: {: <5}, loss: {: .3f}'.format(
                            e, idx, x.item()))
                loss = x / grad_acc
                loss.backward()
                if (idx + 1) % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        optimizer.zero_grad()
        # Save lora - start
        adapter_state_dict = {k: v.cpu() for k, v in self.adapter_params.items()}
        #print(adapter_state_dict)
        logger.info(os.getcwd())
        torch.save(adapter_state_dict, 'adapter_{}-VH-VL_aa-pad.pt'.format(num_epochs))
        # Save lora - end

        #logger.info(f'Final designed sequences:')
        #for seq in self.decode(self.x_seqs):
        #    logger.info(seq)
        #self.output_seq = self.decode(self.x_seqs)[0]


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    args_no_spaces = [arg.replace(" ", "") for arg in sys.argv[1:]]
    logger.info(f"Running with args: {' '.join(args_no_spaces)}")

    start_time = time.time()
    des = Designer(cfg)

    des.run_from_cfg()
    logger.info("finished after %s hours", (time.time() - start_time) / 3600)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
