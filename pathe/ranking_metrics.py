from typing import Optional, Dict

from torchmetrics import Metric
import torch
from collections import defaultdict


SMALLEST_FLOAT = -torch.finfo(torch.float32).max


class RelationMRR(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, filter_dict: Dict):  # dataset: str, tokens_to_idxs: Dict
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.tokens_to_idxs = tokens_to_idxs
        # self.all_relations = defaultdict(set)
        # self.make_all_relation_dict(dataset)
        self.all_relations = filter_dict

    # def make_all_relation_dict(self, dataset):
    #     """
    #     Creates a dictionary which maps each head and tail to a set of the
    #     ground truth relations between them. To be used for filtering during
    #     metric calculation.
    #     Returns
    #     -------
    #
    #     """
    #     triple_db = KgLoader(dataset=dataset, add_inverse=False)
    #     all_triples = torch.cat((triple_db.train_triples,
    #                              triple_db.val_triples,
    #                              triple_db.test_triples), dim=0)
    #     for i in range(all_triples.size()[0]):
    #         self.all_relations[(all_triples[i, 0].item(),
    #                             all_triples[i, 2].item())].add(
    #             self.tokens_to_idxs[all_triples[i, 1].item()])

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------"""

        for i in range(scores.size()[0]):
            head = triples[i, 0].item()
            tail = triples[i, 1].item()
            relation = triples[i, 2].item()
            # filtering out all relations that may be true but are not the
            # one we are predicting and may increase the rank of the correct
            # answer since they may also be given a high score by the model
            # the -1.0 is arbitrary, may need to be increased to increase
            # the rank of filtered relations depending on the magnitude of
            # the scores
            for j in self.all_relations[(head, tail)] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        target_relations = triples[:, 2]
        # subtract the target relation ID (index) from all indexes, making
        # the cell containing the ID (index) of the true relation zero
        # sorted_indices -= np.expand_dims(relations, 1)
        target_relations = target_relations.unsqueeze(1)
        correct_values_to_zero = sorted_indices - target_relations
        # Get the coordinates of each zero (row, column)
        zero_indices = (correct_values_to_zero == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1
        self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.reciprocal_ranks / self.total


class RelationMR(Metric):
    """
    A torchmetric implementation of the Mean Rank for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = False

    def __init__(self, filter_dict: Dict):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("ranks", default=torch.tensor(0, dtype=
        torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.tokens_to_idxs = tokens_to_idxs
        # self.all_relations = defaultdict(set)
        # self.make_all_relation_dict(dataset)
        self.all_relations = filter_dict

    # def make_all_relation_dict(self, dataset):
    #     """
    #     Creates a dictionary which maps each head and tail to a set of the
    #     ground truth relations between them. To be used for filtering during
    #     metric calculation.
    #     Returns
    #     -------
    #
    #     """
    #     triple_db = KgLoader(dataset=dataset, add_inverse=False)
    #     all_triples = torch.cat((triple_db.train_triples,
    #                              triple_db.val_triples,
    #                              triple_db.test_triples), dim=0)
    #     for i in range(all_triples.size()[0]):
    #         self.all_relations[(all_triples[i, 0].item(),
    #                             all_triples[i, 2].item())].add(
    #             self.tokens_to_idxs[all_triples[i, 1].item()])

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """
        for i in range(scores.size()[0]):
            head = triples[i, 0].item()
            tail = triples[i, 1].item()
            relation = triples[i, 2].item()
            # filtering out all relations that may be true but are not the
            # one we are predicting and may increase the rank of the correct
            # answer since they may also be given a high score by the model
            # the -1.0 is arbitrary, may need to be increased to increase
            # the rank of filtered relations depending on the magnitude of
            # the scores
            for j in self.all_relations[(head, tail)] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        target_relations = triples[:, 2]
        # subtract the target relation ID (index) from all indexes, making
        # the cell containing the ID (index) of the true relation zero
        # sorted_indices -= np.expand_dims(relations, 1)
        target_relations = target_relations.unsqueeze(1)
        correct_values_to_zero = sorted_indices - target_relations
        # Get the coordinates of each zero (row, column)
        zero_indices = (correct_values_to_zero == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1
        self.ranks += torch.sum(ranks.float(), dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.ranks / self.total


class RelationHitsAtK(Metric):
    """
    A torchmetric implementation of Hits@K for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, filter_dict: Dict, k: int):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("hits", default=torch.tensor(0, dtype=
        torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.tokens_to_idxs = tokens_to_idxs
        # self.all_relations = defaultdict(set)
        # self.make_all_relation_dict(dataset)
        self.all_relations = filter_dict
        self.k = k

    # def make_all_relation_dict(self, dataset):
    #     """
    #     Creates a dictionary which maps each head and tail to a set of the
    #     ground truth relations between them. To be used for filtering during
    #     metric calculation.
    #     Returns
    #     -------
    #
    #     """
    #     triple_db = KgLoader(dataset=dataset, add_inverse=False)
    #     all_triples = torch.cat((triple_db.train_triples,
    #                              triple_db.val_triples,
    #                              triple_db.test_triples), dim=0)
    #     for i in range(all_triples.size()[0]):
    #         self.all_relations[(all_triples[i, 0].item(),
    #                             all_triples[i, 2].item())].add(
    #             self.tokens_to_idxs[all_triples[i, 1].item()])

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """

        Parameters
        ----------
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------

        """
        for i in range(scores.size()[0]):
            head = triples[i, 0].item()
            tail = triples[i, 1].item()
            relation = triples[i, 2].item()
            # filtering out all relations that may be true but are not the
            # one we are predicting and may increase the rank of the correct
            # answer since they may also be given a high score by the model
            # the -1.0 is arbitrary, may need to be increased to increase
            # the rank of filtered relations depending on the magnitude of
            # the scores
            for j in self.all_relations[(head, tail)] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        target_relations = triples[:, 2]
        # subtract the target relation ID (index) from all indexes, making
        # the cell containing the ID (index) of the true relation zero
        # sorted_indices -= np.expand_dims(relations, 1)
        target_relations = target_relations.unsqueeze(1)
        correct_values_to_zero = sorted_indices - target_relations
        # Get the coordinates of each zero (row, column)
        zero_indices = (correct_values_to_zero == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1
        # hits = torch.mean((ranks <= k), dtype=torch.float)
        self.hits += torch.sum((ranks <= self.k), dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.hits / self.total


class EntityMRR(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, triples: torch.IntTensor, scores: torch.tensor,
               num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """

        triples_per_sample = num_entities_per_sample + 1
        unstacked_triples = triples.reshape(triples.size()[
                                                0] // triples_per_sample,
                                            triples_per_sample, -1)

        scores_unpacked = scores.reshape(triples.size()[
                                             0] // triples_per_sample,
                                         triples_per_sample, -1)

        # get the scores of the correct relation for each triple. For each h,r
        # we generate all possible (h,r,t) pairs and thus the score of the
        # (h,r,t)|(h,t) is the score of the head and the correct relation.
        # Thus, the score for each triple is the score produced by the
        # relation predictor for the relation.
        triple_scores = scores_unpacked[torch.arange(scores_unpacked.size()[
                                                         0]), :,
                        unstacked_triples[:, 0, 2]]  ##### 2 for other format

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(triple_scores, dim=1, descending=True)

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1

        self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.reciprocal_ranks / self.total


class EntityHitsAtK(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for entity
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, k: int):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("hits", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.k = k

    def update(self, triples: torch.IntTensor, scores: torch.tensor,
               num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """

        triples_per_sample = num_entities_per_sample + 1
        unstacked_triples = triples.reshape(triples.size()[
                                                0] // triples_per_sample,
                                            triples_per_sample, -1)

        scores_unpacked = scores.reshape(triples.size()[
                                             0] // triples_per_sample,
                                         triples_per_sample, -1)

        # get the scores of the correct relation for each triple. For each h,r
        # we generate all possible (h,r,t) pairs and thus the score of the
        # (h,r,t)|(h,t) is the score of the head and the correct relation.
        # Thus, the score for each triple is the score produced by the
        # relation predictor for the relation.
        triple_scores = scores_unpacked[torch.arange(scores_unpacked.size()[
                                                         0]), :,
                        unstacked_triples[:, 0, 2]]  ##### 2 for other format

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(triple_scores, dim=1, descending=True)

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1
        self.hits += torch.sum((ranks <= self.k), dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.hits / self.total
