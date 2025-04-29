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
import scipy

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

from utils.sampling import (
    set_rng_seeds,)

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
        #lora_missing, lora_unexpected = self.LM.load_state_dict(
        #    torch.load(os.path.join('..', '..', '..',
        #                            'adapter_512-VH-VL_aa.pt'),
        #               map_location="cpu",
        #               weights_only=True),
        #    strict=False)
        #assert all('lora' not in x for x in lora_missing)
        #assert len(lora_unexpected) == 0
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

    def train_eval(self, train_split, val_split, optimizer, mse, seq_max_len):
        from affinity_dataset import AffinityDataset
        train_ds, val_ds = [
            AffinityDataset(
                dataset=df,
                batch_converter=self.vocab.get_batch_converter(),
                sep_str="<sep>",
                max_len=seq_max_len,
                seed=self.seed + idx,
            ) for idx, df in enumerate([train_split, val_split])
        ]
        #num_epochs = 512 * 10
        num_epochs = 10
        batch_size = 1
        grad_acc = 16
        train_dataloader, val_dataloader = [
            torch.utils.data.DataLoader(ds,
                                        batch_size=batch_size,
                                        shuffle=idx == 0)
            for idx, ds in enumerate([train_ds, val_ds])
        ]

        def f(epoch):
            MAE, MSE = [], []
            Corr_ret = []
            self.LM.eval()
            with torch.no_grad():
                for (seq, affinity) in val_dataloader:
                    x = self.LM(
                        seq.cuda(),
                        repr_layers=[self.LM.num_layers],
                    )['representations'][self.LM.num_layers][:, 0, :]
                    pred = self.LM.affinity(x)

                    MAE.append(abs((pred - affinity.cuda()).item()))
                    MSE.append(MAE[-1]**2)
                    Corr_ret.append((pred.item(), affinity.item()))
                MAE = np.asarray(MAE).mean()
                MSE = np.asarray(MSE).mean()
                Corr_ret = np.asarray(Corr_ret)
                Corr = scipy.stats.pearsonr(
                    Corr_ret[:, 0],
                    Corr_ret[:, 1],
                ).statistic
            logger.info(
                "epoch: {}\nMAE: {: .2f}%, MSE: {: .2f}%, Corr: {: .3f}".format(
                    epoch,
                    MAE * 100,
                    MSE * 100,
                    Corr,
                ))
            self.LM.train()
            return (MAE, MSE, Corr)

        self.LM.train()
        for e in range(num_epochs):
            # SET_EPOCH!!!
            train_ds.set_epoch(e)
            for idx, (seq, affinity) in enumerate(train_dataloader):
                x = self.LM(
                    seq.cuda(),
                    repr_layers=[self.LM.num_layers],
                )['representations'][self.LM.num_layers][:, 0, :]
                pred = self.LM.affinity(x)
                x = mse(pred, affinity.cuda())

                if random.uniform(0, 1) < 1e-2:
                    logger.info(
                        'epoch: {: <5}, idx: {: <5}, loss: {: .3f}'.format(
                            e, idx, x.item()))
                loss = x / grad_acc
                loss.backward()
                if (idx + 1) % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            f(e)
            #if (e + 1) % (512 * 2) == 0:
            #    optimizer.zero_grad()
            #    # Save lora - start
            #    adapter_state_dict = {
            #        k: v.cpu() for k, v in self.adapter_params.items()
            #    }
            #    #print(adapter_state_dict)
            #    logger.info(os.getcwd())
            #    #torch.save(adapter_state_dict,
            #    #           'adapter-epoch-{}-affinity.pt'.format(e + 1))
            #    # Save lora - end
        #logger.info(f'Final designed sequences:')
        #for seq in self.decode(self.x_seqs):
        #    logger.info(seq)
        #self.output_seq = self.decode(self.x_seqs)[0]
        return f(num_epochs)

    def run_from_cfg(self):
        import pandas as pd
        logger.info(os.getcwd())

        # Read the CSV file
        prefix = '/home/zhaoxin/esm/examples/lm-design/affinity-data'
        df = pd.read_csv(os.path.join(prefix, 'SKEMPI.csv'))
        label2seq = {}
        with open(os.path.join(prefix, 'SKEMPI_seq.txt'), 'r') as file:
            for line in file:
                key, value = line.strip().split(None, 1)
                label2seq[key] = value
        for _ in ['A', 'B']:
            df[_] = df[_].map(label2seq)
        df['Affinity'] = (df['Affinity'] - df['Affinity'].min()) / (
            df['Affinity'].max() - df['Affinity'].min()) if df['Affinity'].max(
            ) != df['Affinity'].min() else 0.0
        assert not (df['A'].isnull().any() or df['B'].isnull().any())
        seq_max_len = max([df[_].str.len().max() for _ in ('A', 'B')])

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        folds = []
        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            folds.append((train_df, val_df))
        # optimizer
        for name, param in self.LM.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # lezhang.thu - start
        self.LM.affinity = torch.nn.Sequential(
            torch.nn.Linear(self.LM.embed_dim, 1),
            torch.nn.Sigmoid(),
        ).cuda()
        # lezhang.thu - end

        lora_params = [
            param for name, param in self.LM.named_parameters()
            if 'lora' in name
        ]
        result = []
        mse = torch.nn.MSELoss()
        from itertools import chain
        for (train_df, eval_df) in folds:
            # re-initialize for every fold
            for param in chain(lora_params, self.LM.affinity.parameters()):
                if param.requires_grad:
                    if param.dim() >= 2:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
            result.append(
                self.train_eval(
                    train_df,
                    val_df,
                    torch.optim.AdamW(
                        chain(lora_params, self.LM.affinity.parameters()),
                        lr=1e-4,
                        weight_decay=0.01,
                    ),
                    mse,
                    seq_max_len,
                ))
        result = np.asarray(result)
        t = result.mean(0)
        logger.info(
            "Aggregate\nMAE: {: .2f}%, MSE: {: .2f}%, Corr: {: .3f}".format(
                t[0] * 100,
                t[1] * 100,
                t[2],
            ))


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
