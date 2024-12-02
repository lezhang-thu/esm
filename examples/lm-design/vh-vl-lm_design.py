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
                                    'adapter_512-VH-VL_aa-hyphen.pt'),
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

        K = len(self.vocab)
        #AA_indices = torch.arange(K, device=self.device)[self.vocab_mask_AA]
        # debug
        #x = "EVQLVESGGGLVQPGGSLRLSCAVSGFTVSRNYMTWARQAPGKGPEWVSVIYPGGSTFYADSVKGRFTISRDSSKNILYLQMNSLRVDDTAVYYCARDLTIRGEASWGQGTLVTVSS"
        #x = ["<mask>"] * self.L
        x_vh = 'EVQLVESGGGLIQPGGSLRLSCAASGFIVSRNYMNWVRQAPGKGLEWVALIYSGGSTFYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLVVYGLDYWGQGTLVTVSS'
        x_vl = 'AIQLTQSPSSLSASVGDRVTITCRASQGISSHLAWYQQKPGKAPKLLIFAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQHLNSNPPITFGQGTRLEIK'
        start_seq = list(x_vh + x_vl)
        x = ['-'] * (self.vh_max_len + self.vL_max_len)
        x[:len(x_vh)] = start_seq[:len(x_vh)]
        x[self.vh_max_len:self.vh_max_len + len(x_vl)] = start_seq[len(x_vh):]

        x_seq = []
        import random
        for _ in x:
            #if random.random() < .0:
            #if random.random() < 0.5:
            if random.random() < 0.2:
                x_seq.append('<mask>')
            else:
                x_seq.append(_)

        self.x_seqs = self.encode([x_seq], onehot=True)
        assert self.x_seqs.shape == (self.B, self.L, K)
        return
        #bt = torch.from_numpy(
        #    np.random.choice(AA_indices.cpu().numpy(),
        #                     size=(B, self.L))).to(self.device)
        #self.x_seqs = F.one_hot(
        #    torch.full((
        #        self.B,
        #        self.L,
        #    ),
        #               self.vocab.mask_idx,
        #               device=self.device), K).float()
        #self.x_seqs = self.antibody(self.x_seqs)['logits'].argmax(-1)
        #self.x_seqs = F.one_hot(self.x_seqs, K).float()
        self.x_seqs = F.one_hot(bt, K).float()
        for _ in self.decode(self.x_seqs):
            print(_)
        #self.init_seqs = self.x_seqs.clone()

    ##########################################
    # Losses
    ##########################################
    def calc_x(
        self,
        x_seqs,
    ):
        #return lm_marginal(self.antibody, x_seqs, mask=None)
        # VERY IMPORTANT FOR **mask**
        return lm_marginal(self.antibody,
                           x_seqs,
                           mask=x_seqs.argmax(-1, keepdim=True)
                           != self.vocab.padding_idx)

    def calc_mlm(
        self,
        x_seqs,
    ):
        return self.antibody(x_seqs)

    def calc_sequence_loss(self,
                           x_seqs,
                           LM_losses={'CE_x_pLM': 1.0},
                           mask=None):
        """
        Calculate pLM (LM output probabilities) based on mask-1-out over all positions.
        Calculate seq_losses and combine according to weights in LM_losses.
        Args:
            x_seqs (torch.float32): [B, L, K]
        Returns:
            LM_loss (torch.float32): [B]
            LM_out_logprobs (torch.float32): [B, L, K]
            logging_dict: {other_metrics: torch.float32 [B]}
        """
        B, L, K = x_seqs.shape
        if mask is None:
            mask = torch.ones(B, L, 1, device=self.device).bool()
        n = assert_valid_mask(mask, x_seqs)

        # debug
        #LM_out_logprobs = lm_marginal(self.LM, x_seqs, mask=mask)
        #LM_out_logprobs = lm_marginal(self.antibody, x_seqs, mask=mask)
        #LM_out_logprobs = lm_marginal(self.LM, x_seqs, mask=mask)
        # debug - start
        for _ in range(1000):
            LM_out_logprobs = lm_marginal(self.antibody, x_seqs, mask=None)
            #print('x_seqs.shape: {}'.format(x_seqs.shape))
            #print('LM_out_logprobs.shape: {}'.format(LM_out_logprobs.shape))
            #print('n: {}'.format(n))
            #exit(0)
        exit(0)
        # debug - end

        # For loss calculations, only calculate based on the portion in `mask`.
        x_seqs_masked = x_seqs.masked_select(mask).reshape(B, n, K)
        losses = {
            'CE_x_pLM': -(x_seqs_masked * LM_out_logprobs).sum(-1).mean(-1),
        }
        LM_loss = sum(w * losses[name] for name, w in LM_losses.items())

        return LM_loss, LM_out_logprobs, losses

    def calc_ngram_loss(self, x_seqs, ngram_orders=[1, 2, 3, 4]):
        B = x_seqs.size(0)
        ngram_loss = torch.zeros(B).to(x_seqs)
        seqs = self.decode(x_seqs)
        for order in ngram_orders:
            for i in range(len(seqs)):
                ngram_loss[i] += ngram_utils.compute_kl_div(seqs[i], order)
        return ngram_loss  # [B]

    def calc_structure_loss(self, x_seq, temp_struct=None):
        """Maps x_seq to the structure loss"""
        B, L, K = x_seq.shape

        res_preds = self.struct_model(x_seq)

        if temp_struct is not None:
            # Apply temp to res_preds output
            for coord in COORDS4D_NAMES:
                res_preds[f'{coord}_logits'] /= temp_struct
                res_preds[f'p_{coord}'] = res_preds[f'{coord}_logits'].softmax(
                    -1)

        # Mask handling
        mask = torch.ones_like(self.coords[:, 0, :, :]).bool()
        target_pos_mask = self.target_contacts[None]  # [1, L, L]
        target_neg_mask = self.target_no_contacts[None]  # [1, L, L]
        target_pos_mask &= mask
        target_neg_mask &= mask
        # The below is also: self.valid_contacts & mask
        target_all_mask = target_pos_mask | target_neg_mask

        loss_dict = {}
        targets = ['dist']
        if not self.target_no_angles:
            targets += COORDS_ANGLE_NAMES
        for i, targetname in enumerate(targets):
            target = self.coords[:, i, :, :]
            if target.size(0) == 1:
                res_preds_B = res_preds['p_dist'].shape[0]
                target = target.repeat(res_preds_B, 1, 1)
            else:
                assert_shape(target, B, L, L)

            loss_dict[f'{targetname}_cce'] = get_cce_loss(
                res_preds[f'p_{targetname}'], target, target_all_mask)
            cce_pos = get_cce_loss(res_preds[f'p_{targetname}'], target,
                                   target_pos_mask)
            cce_neg = get_cce_loss(res_preds[f'p_{targetname}'], target,
                                   target_neg_mask)
            loss_dict[f'{targetname}_cce_norm_avg'] = (cce_pos + cce_neg) / 2
            loss_dict[f'{targetname}_cce_pos'] = cce_pos

        CHOSEN_LOSSES = ['dist_cce_pos']  # Worked best in our experiments
        total_loss = sum(loss_dict[k] for k in CHOSEN_LOSSES)

        return total_loss, loss_dict

    def calc_total_loss(self,
                        x,
                        mask,
                        LM_w,
                        struct_w,
                        ngram_w,
                        ngram_orders,
                        temp_struct=None):
        #"""
        #Easy one-stop-shop that calls out to all the implemented loss calculators,
        #aggregates logs, and weights total_loss.

        #As a refresher:
        #    calc_sequence_loss:
        #        calculates \sum log p(x_i|x_\i) for i in {set bits in mask}.
        #            If mask is all ones, this is equal to Pseudo-log-likelihood.
        #        NOTE: every position in mask is masked *separately*
        #            Therefore, there will be multiple forward passes of the LM.
        #    calc_structure_loss:
        #        calculates p(y|x)
        #    calc_ngram_loss:
        #        calculates p_ngram(x)
        #"""

        if mask is not None:
            assert_valid_mask(mask, x=x)
        logs = {}
        total_loss = torch.zeros(x.size(0)).to(x)
        if LM_w:
            lm_m_nlls, _, lm_loss_dict = self.calc_sequence_loss(x, mask=mask)
            lm_m_nlls *= LM_w / self.L
            total_loss += lm_m_nlls
            logs['lm_loss'] = lm_m_nlls
            logs.update(lm_loss_dict)
        #if struct_w:
        if False:
            struct_m_nlls, struct_loss_dict = self.calc_structure_loss(
                x, temp_struct=temp_struct)
            struct_m_nlls *= struct_w
            total_loss += struct_m_nlls
            logs['struct_loss'] = struct_m_nlls
            logs.update(struct_loss_dict)
        #if False:
        if ngram_w:
            ngram_m_nlls = self.calc_ngram_loss(x, ngram_orders=ngram_orders)
            ngram_m_nlls *= ngram_w
            total_loss += ngram_m_nlls
            logs['ngram_loss'] = ngram_m_nlls

        return total_loss, logs  # [B], Dict[str:[B]]

    ##########################################
    # YAML Execution
    ##########################################

    def run_from_cfg(self):
        """
        Main run-loop for the Designer. Runs a relevant design procedure from the config.
        """
        if False:
            self.recover_seq()
            exit(0)
        logger.info(f'Designing sequence for task: {self.cfg.task}')

        design_cfg = self.cfg.tasks[self.cfg.task]
        if self.cfg.task == 'fixedbb':
            stage_fixedbb(self, design_cfg)
        elif self.cfg.task == 'free_generation':
            #stage_free_generation(self, **design_cfg)
            stage_free_generation(self, logger, **design_cfg)
        else:
            raise ValueError(f'Invalid task: {self.cfg.task}')

        logger.info(f'Final designed sequences:')
        for seq in self.decode(self.x_seqs):
            logger.info(seq)
        self.output_seq = self.decode(self.x_seqs)[0]

    @torch.no_grad
    def recover_seq(self):
        import copy
        import re
        import random
        import pandas as pd
        import pickle

        batch_converter = self.antibody.vocab.get_batch_converter()
        # Read the CSV file
        x = list(
            set(
                list(
                    pd.read_csv(
                        "/home/ubuntu/lezhang.thu/biology-research/covid/esm/examples/lm-design/train-vh-vl-aa.csv"
                    )["VL_aa"])) -
            set(
                list(
                    pd.read_csv(
                        "/home/ubuntu/lezhang.thu/biology-research/covid/esm/examples/lm-design/antibody9_16.csv"
                    )["VL_aa"])))
        x.sort()
        diff_seq = list()
        for _ in x:
            if '*' not in _:
                diff_seq.append(_)
        t = copy.deepcopy(diff_seq)

        # mask - start
        def mask_string(text, mask_token="<mask>", mask_prob=0.15):
            masked_text = []
            for char in text:
                # Decide to mask this character based on the mask probability
                if random.random() < mask_prob:
                    masked_text.append(mask_token)
                else:
                    masked_text.append(char)
            return ''.join(masked_text)

        for idx, _ in enumerate(diff_seq):
            diff_seq[idx] = mask_string(_)
        ## debug - start
        #if True:
        #    with open(os.path.join('..', '..', '..', 'result-0.pkl'),
        #              'rb') as f:
        #        x = pickle.load(f)
        #        for idx, _ in enumerate(x):
        #            assert diff_seq[idx] == _
        #        print('all true')
        #        exit(0)
        #with open(os.path.join('..', '..', '..', 'result-0.pkl'), 'wb') as f:
        #    pickle.dump(diff_seq, f)
        #exit(0)
        ## debug - end
        # mask - end

        data = []
        for idx, _ in enumerate(diff_seq):
            data.append(("protein{}".format(idx), _))
        counter_match = 0
        for k in range(len(data) // 32):
            tgt = t[k * 32:min((k + 1) * 32, len(t))]
            x = data[k * 32:min((k + 1) * 32, len(data))]
            _, _, batch_tokens = batch_converter(x)
            batch_tokens = batch_tokens.to(self.device)
            x = self.antibody.model(batch_tokens)["logits"][
                batch_tokens == self.vocab.mask_idx].argmax(-1)
            batch_tokens[batch_tokens == self.vocab.mask_idx] = x
            for idx, seq in enumerate(self.decode(batch_tokens, False)):
                seq = re.search(r"<cls>(.*?)<eos>", seq).group(1)
                if seq == tgt[idx]:
                    counter_match += 1
                else:
                    pass
                    #print('#' * 20)
                    #print(seq)
                    #print(tgt[idx])
                #print('counter_match: {}'.format(counter_match))
        print('counter_match: {}, total: {}'.format(counter_match, len(t)))
        #data = [
        #    ("protein0",
        #     "<mask>VQLVESGGG<mask>IQPGGSLRLSCAASGFIVSRNYM<mask>WVRQAPGKGLEWVALIYSGGSTFYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLVVYGLDYWGQGTLVTVSS"
        #    ),
        #    ("protein1",
        #     "EVQLLESGGGLIQP<mask>GSLRLSCAASGVTVSSNYMSWVRQAPGKGLEWVSL<mask>FAGGSTFYADSVRGR<mask>TISRDNSKNTLYLQMNTLR<mask>EDTAMYFCARDLGPAGATDCWGQGTLVTVSS"
        #    ),
        #    ("protein2",
        #     "QVQLVESGGGLIQPGGS<mask>RLSCAASEFIVSKNYMS<mask>VRQAPGKGLEWVSVIYPGGTTYYA<mask>SVRGRFTI<mask>RDNSKNTLYLQMNSLRAEDTAVYYCARDYGDFYFDYWGQGTLVTVSS"
        #    ),
        #    ("protein3",
        #     "EVQLVESGGGLVQP<mask>GSLRLSCAASEIIVSSNY<mask>SWVRQAPGMGLEWVSVVYSGGSTFYADS<mask>KGRFTISRHNSKNTLYLQMNSLRTEDT<mask>VYYCAREAPNSRGSGT<mask>FDYWG<mask>GTLVTVSS"
        #    ),
        #    ("protein4",
        #     "EVQLVESGG<mask>LIQPGGSLR<mask>SCAASGFTVSSNYMSWVRQAPKKGL<mask>WVSVIYSGGSTYYADSVKGRFTI<mask>RDNSKNTLYLQMNSLRAEDTAVY<mask>CAREGAAANTHGWFDPWG<mask>GTLVTVSS"
        #    ),
        #]

        #_, _, batch_tokens = batch_converter(data)
        #batch_tokens = batch_tokens.to(self.device)
        ##print('self.antibody.model(batch_tokens)["logits"].shape: {}'.format(self.antibody.model(batch_tokens)["logits"].shape))
        ##print('(batch_tokens == self.vocab.mask_idx).shape: {}'.format((batch_tokens == self.vocab.mask_idx).shape))
        ##exit(0)
        #x = self.antibody.model(batch_tokens)["logits"][
        #    batch_tokens == self.vocab.mask_idx].argmax(-1)
        #batch_tokens[batch_tokens == self.vocab.mask_idx] = x
        #for seq in self.decode(batch_tokens, False):
        #    logger.info(seq)

    def init_schedulers_from_cfg(self, cfg: DictConfig):
        """
        Similar to init_schedulers, but expects a stage-specific DictConfig.
        Populates self.schedulers with dotlist key.
        (Simplifies later OmegaConf accesses)
        Example:
        cfg = {
            num_iter: 10,
            sub_cfg: {
                my_sched: {
                    scheduler: CosineAnnealingLR
                    initial: 1e-2
                    T_max: 200}}}
        Effect:
            self.schedulers['sub_cfg.my_sched'] = <Scheduler>
        """

        def walk_cfg(d, parent_key='', sep='.'):
            from collections.abc import MutableMapping
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                yield (new_key, v)
                if isinstance(v, MutableMapping):
                    yield from walk_cfg(v, new_key, sep=sep)

        from typing import Optional, Dict, List, Any, Union

        def is_sspec(maybe_sspec: Union[SchedulerSpec, Any]):
            infer_from_key = (isinstance(maybe_sspec, DictConfig) and
                              maybe_sspec.get('scheduler', None) is not None)
            # infer_from_type = OmegaConf.get_type(maybe_sspec) is SchedulerSpec
            return infer_from_key

        if not self.resuming_stage:
            for name, maybe_sspec in walk_cfg(cfg, sep='.'):
                if is_sspec(maybe_sspec):
                    assert not name in self.schedulers, f"Trying to re-register {name}"
                    self.schedulers[name] = to_scheduler(maybe_sspec)

    def gen_step_cfg(self, cfg):
        """
        Replace schedulers in a cfg with step-specific values.
        Make sure to call `init_schedulers_from_cfg(cfg)` first!
        Uses Designer state:
            - self.schedulers
        """
        step_cfg = cfg.copy()
        for name, sched in self.schedulers.items():
            if OmegaConf.select(step_cfg, name) is not None:
                OmegaConf.update(step_cfg, name, sched(), merge=False)
        return step_cfg

    def stepper(self, iterable, update_schedulers=True, cfg=None):
        self.init_schedulers_from_cfg(cfg)

        for local_step in iterable:
            yield local_step, self.gen_step_cfg(cfg)

            if update_schedulers:
                self.update_schedulers()

    def update_schedulers(self):
        for s in self.schedulers.values():
            try:
                s.step()
            except AttributeError:
                pass  # constants: dummy lambda

    def init_schedulers(self, **kwargs):
        """
        Schedulers (always stage-specific) are initialized according to SchedulerSpec,
        and depend on global_step
        Optionally wrapping an optimizer class with single param group.
        Stores the schedulers in self._schedulers
        Returns:
            functions which return the current value for each
        """
        set_scheduler_repo(self.cfg.get('schedulers', {}))
        for name, sspec in kwargs.items():
            assert not name in self.schedulers, f"Trying to re-register {name}"
            self.schedulers[name] = to_scheduler(sspec)
        assert sys.version_info >= (
            3, 6), "py>=3.6 preserve kwarg and dict order see PEP468"
        return [self.schedulers[name] for name in kwargs]


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
