"""
Scalable routines for training and finetuning a PathER model.

"""

import logging
from functools import partial

import torch
from torch import nn
import pandas as pd
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

from pathe_ranking_metrics import (RelationMRR, RelationHitsAtK,
                                   EntityMRR, EntityHitsAtK)


logger = logging.getLogger(__name__)


def debug_single_path(batch, trim_paths: bool=False, override_pos: bool=False):
    if trim_paths:
        path_origins = batch['net_input']['path_origins']
        new_path_origins, counts = torch.unique_consecutive(path_origins,
                                                            return_counts=True)
        ccounts = torch.cumsum(counts, dim=0)
        ccounts_clone = ccounts.clone()
        ccounts[1:] = ccounts_clone[:-1]
        ccounts[0] = 0
        new_ppt = torch.ones_like(batch["ppt"]) * 2
        new_ent_paths = batch['net_input']['ent_paths'][ccounts, :]
        new_rel_paths = batch['net_input']['rel_paths'][ccounts, :]
        new_ent_lengths = batch['net_input']['ent_lengths'][ccounts]
        new_head_idxs = batch['net_input']['head_idxs'][ccounts]
        new_tail_idxs = batch['net_input']['tail_idxs'][ccounts]
        new_pos = batch['net_input']['pos'][ccounts, :]

        batch['ppt'] = new_ppt
        batch['net_input']['ent_paths'] = new_ent_paths
        batch['net_input']['rel_paths'] = new_rel_paths
        batch['net_input']['ent_lengths'] = new_ent_lengths
        batch['net_input']['head_idxs'] = new_head_idxs
        batch['net_input']['tail_idxs'] = new_tail_idxs
        batch['net_input']['pos'] = new_pos
        batch['net_input']['path_origins'] = new_path_origins
    if override_pos:
        device = batch['net_input']['pos'].device
        pos_size = batch['net_input']['pos'].size()
        new_pos = torch.arange(1, pos_size[1]+1, device=device).repeat(
            pos_size[0], 1)
        new_pos[(batch['net_input']['pos'] == 0)] = 0
        previous = batch['net_input']['pos']
        batch['net_input']['pos'] = new_pos
        del(previous)


    return batch



def pathe_forward_step(model, criterion, batch, **kwargs):
    """
    PathE forward function for multi-paths relation prediction.
    """
    # batch = debug_single_path(batch, trim_paths=False, override_pos=True)
    ent_paths, rel_paths = \
        batch["net_input"]["ent_paths"], batch["net_input"]["rel_paths"]
    head_idxs = batch["net_input"]["head_idxs"]
    tail_idxs = batch["net_input"]["tail_idxs"]
    entity_origin = batch["net_input"]["path_origins"]
    pos = batch["net_input"]["pos"]
    targets = batch["target"] - 2  # no PAD and MSK in the model output
    # FIXME this is going to change with the actual head-tail idxs from dataset
    head_idxs = head_idxs * 2
    tail_idxs = tail_idxs * 2
    ppt = batch["ppt"]

    logits_rp, logits_lp = model(
        ent_paths=ent_paths,
        rel_paths=rel_paths,
        head_idxs=head_idxs,
        tail_idxs=tail_idxs,
        ppt=ppt, pos=pos,
        entity_origin=entity_origin,
        targets=targets)
    # loss_rp = criterion(logits_rp, batch["target"])
    loss_rp = criterion(logits_rp, targets)
    return logits_rp, loss_rp, logits_lp


class PathEModelWrapper(LightningModule):
    """
    PL Wrapper for training PathE.

    """

    def __init__(self, pathe_model, filtration_dict, num_negatives=0,
                 optimiser="adam", scheduler="none", lr=1e-3, momentum=0,
                 weight_decay=0, class_weights=None, label_smoothing=0.0,
                 train_sub_batch=None, val_sub_batch=None, test_sub_batch=None,
                 val_num_negatives=0, full_test=False, max_ppt=None,
                 margin=10, nssa_alpha=1, lp_loss_fn="nssa",
                 loss_weight: float = 0.5,  **hparams):
        """
        Parameters
        ----------
        pather_model : torch.Module
            A supported Seq2Seq model, with encoder and decoder (e.g. ParthX).
        tokens_to_idxs : dict
            The mapping/encoding from original tokens to their indexes, needed
            for the extraction of the metrics that will be monitored.
        train_cls : bool
            Whether to train the CLS objective on the relation prediction task.

        """
        super().__init__()
        self.model = pathe_model
        self.loss_weight = loss_weight
        self.train_num_negatives = num_negatives
        self.val_num_negatives = val_num_negatives
        self.test_num_negatives = val_num_negatives \
            if not full_test else full_test
        self.class_weights = class_weights
        comp_sub_batch = lambda sub_batch: sub_batch * max_ppt \
            if sub_batch is not None else sub_batch  # CHANGED THIS
        self.train_sub_batch_size = comp_sub_batch(train_sub_batch)
        self.val_sub_batch_size = comp_sub_batch(val_sub_batch) 
        self.test_sub_batch_size = comp_sub_batch(test_sub_batch)
        # Optimisation hyperparams
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.lr = lr  # initial LR
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        # [List of metrics we will be watching on validation and test sets]
        # Relation prediction metrics: always watched
        self.val_relationMRR = RelationMRR(filtration_dict)
        self.val_relationHitsAt1 = RelationHitsAtK(filtration_dict, k=1)
        self.val_relationHitsAt3 = RelationHitsAtK(filtration_dict, k=3)
        self.val_relationHitsAt5 = RelationHitsAtK(filtration_dict, k=5)
        self.val_relationHitsAt10 = RelationHitsAtK(filtration_dict, k=10)
        self.test_relationMRR = RelationMRR(filtration_dict)
        self.test_relationHitsAt1 = RelationHitsAtK(filtration_dict, k=1)
        self.test_relationHitsAt3 = RelationHitsAtK(filtration_dict, k=3)
        self.test_relationHitsAt5 = RelationHitsAtK(filtration_dict, k=5)
        self.test_relationHitsAt10 = RelationHitsAtK(filtration_dict, k=10)
        # self.save_hyperparameters(ignore=['pathe_model'])

        loss_reduction = "mean"
        if self.val_num_negatives > 0:  # watch link prediction metrics
            loss_reduction = 'none'
            self.val_linkMRR = EntityMRR()
            self.val_linkHitsAt1 = EntityHitsAtK(k=1)
            self.val_linkHitsAt3 = EntityHitsAtK(k=3)
            self.val_linkHitsAt5 = EntityHitsAtK(k=5)
            self.val_linkHitsAt10 = EntityHitsAtK(k=10)
            self.test_linkMRR = EntityMRR()
            self.test_linkHitsAt1 = EntityHitsAtK(k=1)
            self.test_linkHitsAt3 = EntityHitsAtK(k=3)
            self.test_linkHitsAt5 = EntityHitsAtK(k=5)
            self.test_linkHitsAt10 = EntityHitsAtK(k=10)

        # Defining losses for CLS objectives
        self.rp_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            reduction=loss_reduction,
            label_smoothing=label_smoothing)
        # self.rp_criterion = torch.nn.MultiMarginLoss(weight=class_weights,
        #                                              margin=margin)
        if lp_loss_fn == "bce":
            self.lp_loss_fn = self.calculate_lp_bce
        elif lp_loss_fn == "ce":
            self.lp_loss_fn = self.calculate_lp_ce
        elif lp_loss_fn == "nssa":
            self.lp_loss_fn = partial(
                self.calculate_lp_nssa, alpha=nssa_alpha, gamma=margin)

        self.model_forward = partial(
            pathe_forward_step, model=self.model,
            criterion=self.rp_criterion)

    def calculate_lp_bce(self, logits, num_negatives):
        """
        Calculates the weighted BCE loss for link prediction.
        Each negative triple is weighted by 1/num negatives.

        Parameters
        ----------
        logits : The logits for each triple with shape (num triple,1)
        num_negatives : The number of negatives for each positive (int)

        """
        # create the labels for the triples as all zeros
        labels = torch.zeros((logits.size()[0]),
                             dtype=torch.float32).to(self.device)
        # make the labels of the true triples 1
        labels[torch.arange(0, labels.size()[0], num_negatives + 1)] = 1.
        # calculate the weight matrix
        downweigh_negs = torch.ones_like(labels) / num_negatives
        # set the weights of true triple to 1
        downweigh_negs[
            torch.arange(0, labels.size()[0], num_negatives + 1)] = 1.
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), labels, reduction="none")
        loss = (loss * downweigh_negs).sum(dim=-1) / downweigh_negs.sum(dim=-1)
        return loss

    def calculate_lp_ce(self, logits, num_negatives):
        """
        Calculates the CE loss for link prediction.

        Parameters
        ----------
        logits : The logits for each triple with shape (num_triples,1)
        num_negatives : The number of negatives
    
        """
        scores_unpacked = logits.reshape(logits.size()[0] //
                                         (num_negatives + 1),
                                         num_negatives + 1)
        labels = torch.zeros((scores_unpacked.size()[0]),
                             dtype=torch.int64, device=self.device)
        loss_lp = torch.mean(nn.functional.cross_entropy(scores_unpacked, labels))
        return loss_lp

    def calculate_lp_nssa(self, scores, num_negs, alpha=1, gamma=10):
        """
        Self-Adversarial Negative Sampling Loss as shown in Sun et al. (RotatE)
        
        Parameters
        ----------
        scores : The logits of the model (num samples, 1)
        num_negs : The number of negatives for each positive sample
        alpha : The adversarial temperature
        gamma : The margin

        """
        score = gamma + scores
        scores_per_sample = num_negs + 1
        unstacked_scores = score.reshape(
            score.size()[0] // scores_per_sample, scores_per_sample, -1)
        
        pos_score = score[torch.arange(0, score.size()[0], num_negs + 1)]
        neg_score = unstacked_scores[torch.arange(unstacked_scores.size()[0]),1:,:]
        neg_score = (nn.functional.softmax(neg_score * alpha, dim=1).detach()
                    * nn.functional.logsigmoid(-neg_score)).squeeze().sum(dim=1)
        pos_score = nn.functional.logsigmoid(pos_score).squeeze()
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        
        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss

    def training_step(self, batch, batch_idx):
        if self.train_sub_batch_size is None:
            logits_rp, loss_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, loss_rp, logits_lp = self.sub_batch(batch=batch,
                                sub_batch_size=self.train_sub_batch_size)
            if self.train_num_negatives == 0:
                loss_rp = torch.mean(loss_rp)

        if self.train_num_negatives > 0:
            # FOR RELATION PREDICTION
            # select the loss that corresponds to the true triples only
            loss_rp = torch.mean(loss_rp[torch.arange(0, loss_rp.size()[0],
                                                   self.train_num_negatives + 1)])
            loss_lp = self.lp_loss_fn(logits_lp, self.train_num_negatives)
            loss = (self.loss_weight * loss_lp) + \
                   ((1.0 - self.loss_weight) * loss_rp)
            self.log("train_loss_rp", loss_rp)
            self.log("train_loss_lp", loss_lp)
            self.log("train_loss", loss, prog_bar=True)
            if torch.isnan(loss):
                logger.warning("Training stopped due to NaN loss.")
                self.trainer.should_stop = True
                self.trainer.limit_val_batches = 0

        else:
            loss = loss_rp
            self.log("train_loss_rp", loss_rp)
            self.log("train_loss", loss_rp, prog_bar=True)

        # make_dot(loss, params=dict(self.model.named_parameters())).render(
        #     "attachedavg", format="png")
        return loss

    def calculate_and_log_val_relation_metrics(self, triples, logits):
        self.val_relationMRR(triples=triples,
                             scores=logits)
        self.val_relationHitsAt1(triples=triples,
                                 scores=logits)
        self.val_relationHitsAt3(triples=triples,
                                 scores=logits)
        self.val_relationHitsAt5(triples=triples,
                                 scores=logits)
        self.val_relationHitsAt10(triples=triples,
                                  scores=logits)
        self.log("valid_mrr", self.val_relationMRR, on_step=False,
                 on_epoch=True)
        self.log("valid_hits1", self.val_relationHitsAt1, on_step=False,
                 on_epoch=True)
        self.log("valid_hits3", self.val_relationHitsAt3, on_step=False,
                 on_epoch=True)
        self.log("valid_hits5", self.val_relationHitsAt5, on_step=False,
                 on_epoch=True)
        self.log("valid_hits10", self.val_relationHitsAt10, on_step=False,
                 on_epoch=True)

    def calculate_and_log_test_relation_metrics(self, triples, logits):
        self.test_relationMRR(triples=triples,
                              scores=logits)
        self.test_relationHitsAt1(triples=triples,
                                  scores=logits)
        self.test_relationHitsAt3(triples=triples,
                                  scores=logits)
        self.test_relationHitsAt5(triples=triples,
                                  scores=logits)
        self.test_relationHitsAt10(triples=triples,
                                   scores=logits)
        self.log("test_mrr", self.test_relationMRR, on_step=False,
                 on_epoch=True)
        self.log("test_hits1", self.test_relationHitsAt1, on_step=False,
                 on_epoch=True)
        self.log("test_hits3", self.test_relationHitsAt3, on_step=False,
                 on_epoch=True)
        self.log("test_hits5", self.test_relationHitsAt5, on_step=False,
                 on_epoch=True)
        self.log("test_hits10", self.test_relationHitsAt10, on_step=False,
                 on_epoch=True)

    def calculate_and_log_val_links_metrics(self, rp_triples, logits_rp):
        self.val_linkMRR(triples=rp_triples, scores=logits_rp,
                         num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt1(triples=rp_triples, scores=logits_rp,
                             num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt3(triples=rp_triples, scores=logits_rp,
                             num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt5(triples=rp_triples, scores=logits_rp,
                             num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt10(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.val_num_negatives)
        self.log("valid_link_mrr", self.val_linkMRR, on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits1", self.val_linkHitsAt1,
                 on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits3", self.val_linkHitsAt3,
                 on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits5", self.val_linkHitsAt5,
                 on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits10", self.val_linkHitsAt10,
                 on_step=False,
                 on_epoch=True)

    def calculate_and_log_test_links_metrics(self, rp_triples, logits_rp):
        self.test_linkMRR(triples=rp_triples, scores=logits_rp,
                          num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt1(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt3(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt5(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt10(triples=rp_triples, scores=logits_rp,
                               num_entities_per_sample=self.test_num_negatives)
        self.log("test_link_mrr", self.test_linkMRR, on_step=False,
                 on_epoch=True)
        self.log("test_link_hits1", self.test_linkHitsAt1,
                 on_step=False,
                 on_epoch=True)
        self.log("test_link_hits3", self.test_linkHitsAt3,
                 on_step=False,
                 on_epoch=True)
        self.log("test_link_hits5", self.test_linkHitsAt5,
                 on_step=False,
                 on_epoch=True)
        self.log("test_link_hits10", self.test_linkHitsAt10,
                 on_step=False,
                 on_epoch=True)



    def sub_batch(self, batch, sub_batch_size):
        """
        A function that sub batches the evaluation so that it fits into GPU
        memory

        Parameters
        ----------
        batch : The batch of network inputs

        """
        # get the sub batch size (in number of paths)
        sub_batch_size = sub_batch_size
        # get the device to send the tensors after accumulation of sub batches
        model_device = self.device
        # get the number of output neurons of the prediction head
        rp_out_size = self.model.rel_embeddings.weight.size()[0] - 2
        # get the size of the batch (in triples)
        batch_size = batch['target'].size()
        # initialize tensors to aggregate the sub_batches
        # these are initialized using the number of triples as we cannot
        # separate the paths belonging to each triple (would mess up the
        # attention and aggregation)
        logits_rp = torch.zeros((batch_size[0], rp_out_size),
                                dtype=torch.float32)
        logits_lp = torch.zeros((batch_size[0]), dtype=torch.float32)
        # loss_lp = torch.zeros(batch_size[0], dtype=torch.float32)
        loss_rp = torch.zeros(batch_size[0], dtype=torch.float32)
        # get the paths per triple
        ppt = batch["ppt"]
        # if the sub_batch size is smaller than the max number of paths per
        # triple, increase it to the max number of paths and print a warning
        # if sub_batch_size < ppt.max():
        #     warnings.warn("Sub-batch size cannot be less than the maximum "
        #                   "paths per "
        #                   "triple, setting sub-batch to: " + str(ppt.max()))
        #     sub_batch_size = ppt.max()
        # initialize indexing variable to get the batches
        # the cumulative sum is used to know how many paths are included when
        # grabbing more than one triple in the subbatch
        ppt_cum_sum = torch.cumsum(ppt, dim=0)
        remaining = ppt_cum_sum
        current_idx_cum_sum = 0
        initial = 0
        initial_triple = 0
        # as long as the remaining paths (last element) is not zero
        while remaining[-1] != 0:
            # start getting triples as long as the number of paths is smaller
            # than the subbatch size
            for i in range(remaining.size()[0]):
                if remaining[i] > sub_batch_size:
                    break
                else:
                    current_idx_cum_sum = i
            # once we found the last index of the triples that can be
            # included in the sub batch create the offsets, so we can slice
            # the batch
            path_offset = remaining[current_idx_cum_sum]
            remaining = remaining - path_offset
            triple_offset = current_idx_cum_sum + 1  # necessary as upper end
            # in slicing is exclusive
            # create the sub batch dictionary
            sub_batch = {
                "id": batch["id"][initial_triple:
                                  triple_offset],
                "ppt": batch["ppt"][initial_triple:
                                    triple_offset],
                "net_input": {
                    "ent_paths": batch["net_input"]["ent_paths"][initial:
                                                                 initial +
                                                                 path_offset,
                                 :],
                    "rel_paths": batch["net_input"]["rel_paths"][initial:
                                                                 initial +
                                                                 path_offset,
                                 :],
                    "ent_lengths": batch["net_input"]["ent_lengths"][initial:
                                                                     initial +
                                                                     path_offset],
                    "head_idxs": batch["net_input"]["head_idxs"][initial:
                                                                 initial +
                                                                 path_offset],
                    "tail_idxs": batch["net_input"]["tail_idxs"][initial:
                                                                 initial +
                                                                 path_offset],
                    "path_origins": batch["net_input"]["path_origins"][initial:
                                                                 initial +
                                                                 path_offset],
                    "pos": batch["net_input"]["pos"][initial:
                                                                 initial +
                                                                 path_offset,
                                 :],
                },
                "target": batch["target"][initial_triple:
                                          triple_offset],
                "ori_triple": batch["ori_triple"][initial_triple:
                                          triple_offset,:],
            }

            # get the model outputs and losses for the sub batch
            sb_logits_rp, sb_loss_rp, sb_logits_lp = self.model_forward(
                batch=sub_batch)
            # aggregate the results in the aggregation tensors
            logits_rp[initial_triple: triple_offset,
            :] = sb_logits_rp
            logits_lp[initial_triple: triple_offset] = sb_logits_lp.squeeze()
            loss_rp[initial_triple: triple_offset] = sb_loss_rp
            # increase the starting indices to slice the next sub_batch
            initial = initial + path_offset
            initial_triple = triple_offset
        return (logits_rp.to(torch.device(model_device)),
                loss_rp.to(torch.device(model_device)),
                logits_lp.to(torch.device(model_device)))






    def validation_step(self, batch, batch_idx):
        if self.val_sub_batch_size is None:
            logits_rp, loss_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, loss_rp, logits_lp = self.sub_batch(batch=batch,
                                    sub_batch_size=self.val_sub_batch_size)
            if self.val_num_negatives == 0:
                loss_rp = torch.mean(loss_rp)

        triples = batch['ori_triple']
        if self.val_num_negatives > 0:
            # FOR RELATION PREDICTION
            # select the logits of the true triple
            logits_rp_only = logits_rp[torch.arange(0, logits_rp.size()[0],
                                                 self.val_num_negatives + 1)]
            # select the true triples
            rp_only_triples = triples[
                torch.arange(0, triples.size()[0],
                             self.val_num_negatives + 1)]
            # select the loss that corresponds to the true triples only
            loss_rp = torch.mean(loss_rp[torch.arange(0, loss_rp.size()[0],
                                                   self.val_num_negatives + 1)])
            # compute and log the losses
            loss_lp = self.lp_loss_fn(logits_lp, self.val_num_negatives)
            loss = (self.loss_weight * loss_lp) + \
                   ((1.0 - self.loss_weight) * loss_rp)
            self.log("valid_loss_rp", loss_rp)
            self.log("valid_loss_lp", loss_lp)
            self.log("valid_loss", loss, prog_bar=True)
            self.calculate_and_log_val_relation_metrics(
                rp_only_triples, logits_rp_only)
            self.calculate_and_log_val_links_metrics(triples, logits_lp)


        else:
            loss = loss_rp
            self.log("valid_loss_rp", loss_rp)
            self.log("valid_loss", loss_rp, prog_bar=True)
            self.calculate_and_log_val_relation_metrics(triples, logits_rp)



    def test_step(self, batch, batch_idx):
        if self.test_sub_batch_size is None:
            logits_rp, loss_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, loss_rp, logits_lp = self.sub_batch(batch=batch,
                                sub_batch_size=self.test_sub_batch_size)
            if self.test_num_negatives == 0:
                loss_rp = torch.mean(loss_rp)

        triples = batch['ori_triple']
        if self.test_num_negatives > 0:
            # FOR RELATION PREDICTION
            # select the logits of the true triple
            logits_rp_only = logits_rp[torch.arange(0, logits_rp.size()[0],
                                                 self.test_num_negatives + 1)]
            # select the true triples
            rp_only_triples = triples[
                torch.arange(0, triples.size()[0],
                             self.test_num_negatives + 1)]
            # select the loss that corresponds to the true triples only
            loss_rp = torch.mean(loss_rp[torch.arange(0, loss_rp.size()[0],
                                                   self.test_num_negatives + 1)])
            # compute and log the losses
            loss_lp = self.lp_loss_fn(logits_lp, self.test_num_negatives)
            loss = (self.loss_weight * loss_lp) + \
                   ((1.0 - self.loss_weight) * loss_rp)
            self.log("test_loss_rp", loss_rp)
            self.log("test_loss_lp", loss_lp)
            self.log("test_loss", loss, prog_bar=True)
            self.calculate_and_log_test_relation_metrics(
                rp_only_triples, logits_rp_only)

            self.calculate_and_log_test_links_metrics(triples, logits_lp)


        else:
            loss = loss_rp
            self.log("test_loss_rp", loss_rp)
            self.log("test_loss", loss_rp, prog_bar=True)
            self.calculate_and_log_test_relation_metrics(triples, logits_rp)




    def configure_optimizers(self):
        """
        Called by PL-Lightning to initialise and setup optimisers and learning
        rate schedulers. Currently supports basic optims and functionalities.

        Note: most of this function can be implemented in a util.
        """
        optim_hps = dict(params=self.model.parameters(),
                         lr=self.lr,  # initial if scheduler
                         weight_decay=self.weight_decay)
        optimiser = None  # one of the supported optimisers below
        if self.optimiser == "adam":
            optimiser = torch.optim.Adam(
                **optim_hps)
        elif self.optimiser == "sgd":
            optimiser = torch.optim.SGD(
                **optim_hps,
                momentum=self.momentum)
        elif self.optimiser == "rms":
            optimiser = torch.optim.RMSprop(
                **optim_hps,
                momentum=self.momentum)
        else:  # not supported or implemented
            raise ValueError(f"Not a valid optimiser: {self.optimiser}")

        opt_dict = {"optimizer": optimiser}
        if self.scheduler != "none":
            raise NotImplementedError()
        return opt_dict  # makes distinction among optimisers and schedulers
