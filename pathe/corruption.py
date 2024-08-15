"""
Corruption utilities for triples and paths, needed for link prediction metrics.

"""
import os
import random
import logging
from typing import Dict
from tqdm import tqdm
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import torch

logger = logging.getLogger("pathe.corruption")


class CorruptTailGenerator:

    def __init__(self, filter_dict: Dict, entities: torch.tensor):
        self.max_index = None
        self.shuffled = None
        self.filter_dict = filter_dict
        self.entities = entities
        self.get_shuffled_indices_tensor()

    def get_shuffled_indices_tensor(self):
        self.max_index = torch.max(self.entities).item() + 1
        self.shuffled = torch.randperm(self.max_index)

    def get_filtered_corrupted_triples(self, triples, k):
        """
        A function that returns k corrupted and filtered triples for each
        triple of the batch provided
        Parameters
        ----------
        triples :The triples to corrupt
        k : the number of corrupted triples to generate for each triple

        Returns
        -------
        A torch tensor of shape (num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities
        max_starting_index = self.max_index - k
        corrupted_triples = []
        for i in range(triples.size()[0]):
            triple = triples[i, :]
            head = triple[0].item()
            relation = triple[1].item()
            # get a random starting index to get the k entities
            start = random.randint(0, max_starting_index)
            candidate_tails = self.shuffled[start:start + k]
            # find if the true tail was sampled or any of the tails that need
            # to be filtered
            entities_to_filter = self.filter_dict[(head, relation)]
            tails_to_filter = candidate_tails
            zero_indices_list = []
            for entity in entities_to_filter:
                tails_to_filter_entity = tails_to_filter - entity
                zero_indices_list.append(
                    (tails_to_filter_entity == 0).nonzero()[:, 0])
            zero_indices = torch.cat(zero_indices_list, dim=0)
            # replace the entities
            # if the starting index was towards the end of the shuffled
            # tensor start sampling replacement candidates before it else
            # start sampling replacements after it
            if start > self.max_index // 2:
                remaining = zero_indices.size()[0]
                while remaining > 0:
                    idx = random.randint(0, start - 1)
                    sample_entity = self.shuffled[idx]
                    if sample_entity.item() not in entities_to_filter:
                        candidate_tails[
                            zero_indices[remaining - 1]] = sample_entity
                        remaining -= 1
            else:
                remaining = zero_indices.size()[0]
                while remaining > 0:
                    idx = random.randint(start + k, self.max_index - 1)
                    sample_entity = self.shuffled[idx]
                    if sample_entity.item() not in entities_to_filter:
                        candidate_tails[
                            zero_indices[remaining - 1]] = sample_entity
                        remaining -= 1
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            hr = triple[0:2]
            # repeat it k times and shape appropriately
            hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted tails
            combined = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                                 dim=0)
            combined = combined.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete = torch.cat((triples[i, :].unsqueeze(0), combined),
                                 dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete)
        return torch.cat(corrupted_triples, dim=0)


class CorruptHeadGenerator:

    def __init__(self, filter_dict: Dict, entities: torch.tensor):
        self.max_index = None
        self.shuffled = None
        self.filter_dict = filter_dict
        self.entities = entities
        self.get_shuffled_indices_tensor()

    def get_shuffled_indices_tensor(self):
        self.max_index = torch.max(self.entities).item() + 1
        self.shuffled = torch.randperm(self.max_index)

    def get_filtered_corrupted_triples(self, triples, k):
        """
        A function that returns k corrupted and filtered triples for each
        triple of the batch provided
        Parameters
        ----------
        triples : The triples to corrupt
        k : the number of corrupted triples to generate for each triple

        Returns
        -------
        A torch tensor of shape (num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities
        max_starting_index = self.max_index - k
        corrupted_triples = []
        for i in range(triples.size()[0]):
            triple = triples[i, :]
            relation = triple[1].item()
            tail = triple[2].item()
            # get a random starting index to get the k entities
            start = random.randint(0, max_starting_index)
            candidate_heads = self.shuffled[start:start + k]
            # find if the true tail was sampled or any of the tails that need
            # to be filtered
            entities_to_filter = self.filter_dict[(relation, tail)]
            heads_to_filter = candidate_heads
            zero_indices_list = []
            for entity in entities_to_filter:
                heads_to_filter_entity = heads_to_filter - entity
                zero_indices_list.append(
                    (heads_to_filter_entity == 0).nonzero()[:, 0])
            zero_indices = torch.cat(zero_indices_list, dim=0)
            # replace the entities
            # if the starting index was towards the end of the shuffled
            # tensor start sampling replacement candidates before it else
            # start sampling replacements after it
            if start > self.max_index // 2:
                remaining = zero_indices.size()[0]
                while remaining > 0:
                    idx = random.randint(0, start - 1)
                    sample_entity = self.shuffled[idx]
                    if sample_entity.item() not in entities_to_filter:
                        candidate_heads[
                            zero_indices[remaining - 1]] = sample_entity
                        remaining -= 1
            else:
                remaining = zero_indices.size()[0]
                while remaining > 0:
                    idx = random.randint(start + k, self.max_index - 1)
                    sample_entity = self.shuffled[idx]
                    if sample_entity.item() not in entities_to_filter:
                        candidate_heads[
                            zero_indices[remaining - 1]] = sample_entity
                        remaining -= 1
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            rt = triple[1:3]
            # repeat it k times and shape appropriately
            rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted heads
            combined = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                                 dim=0)
            combined = combined.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete = torch.cat((triples[i, :].unsqueeze(0), combined),
                                 dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete)
        return torch.cat(corrupted_triples, dim=0)


import data_utils as du



def corrupt_entities(triples, max_starting_index, k, shuffled,
                     head_filter_dict, tail_filter_dict, max_index):
    # The maximum index that we can start sampling from the shuffled
    # tensor and still get k entities
    shuffled_init = shuffled.copy()
    max_starting_index = max_starting_index
    corrupted_triples = []
    for i in tqdm(range(triples.size()[0])):
        triple = triples[i, :]
        head = triple[0].item()
        relation = triple[1].item()
        tail = triple[2].item()
        ##################################
        # Make the tail corruptions #####
        ##################################
        # entities_to_filter = tail_filter_dict[(head, relation)]
        # entities = set(range(max_starting_index))
        # filtered = entities - entities_to_filter
        # candidate_tails_np = np.random.choice(np.array(list(
        #     filtered)), size=k, replace=False).astype(np.int32)
        tensor_ind = random.randint(0, shuffled.shape[0] - 1)
        initial = shuffled[tensor_ind, :].copy()
        # check if initial is a copy
        # assert initial.base is None
        # get a random starting index to get the k entities
        # start = np.random.randint(low=0, high=max_starting_index + 1)
        start = random.randint(0, max_starting_index)
        candidate_tails = shuffled[tensor_ind, start:start + k].copy()
        # assert candidate_tails.base is None
        # find if the true tail was sampled or any of the tails that need
        # to be filtered
        entities_to_filter = tail_filter_dict[(head, relation)].copy()
        # in_d = entities_to_filter.copy()
        tails_to_filter = candidate_tails
        zero_indices_list = []
        for entity in entities_to_filter:
            tails_to_filter_entity = tails_to_filter - entity
            zero_indices_list.append(
                (tails_to_filter_entity == 0).nonzero()[0])
            # if (tails_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
            #     assert entity == candidate_tails.item((
            #                                                   tails_to_filter_entity == 0).nonzero()[
            #                                               0].item())
        zero_indices = np.concatenate(zero_indices_list, axis=0)
        # replace the entities
        # if the starting index was towards the end of the shuffled
        # tensor start sampling replacement candidates before it else
        # start sampling replacements after it
        if start > max_starting_index // 2:
            remaining = zero_indices.shape[0]
            while remaining > 0:
                idx = random.randint(0, start - 1)
                # idx = np.random.randint(0, start)
                sample_entity = shuffled.item((tensor_ind, idx))
                if sample_entity not in entities_to_filter:
                    candidate_tails[
                        zero_indices[remaining - 1]] = sample_entity
                    entities_to_filter.add(sample_entity)
                    remaining -= 1
        else:
            remaining = zero_indices.shape[0]
            while remaining > 0:
                idx = random.randint(start + k, max_index - 1)
                # idx = np.random.randint(start + k, max_index)
                sample_entity = shuffled.item((tensor_ind, idx))
                if sample_entity not in entities_to_filter:
                    candidate_tails[
                        zero_indices[remaining - 1]] = sample_entity
                    entities_to_filter.add(sample_entity)
                    remaining -= 1
        # assert in_d == tail_filter_dict[(head, relation)]
        candidate_tails = torch.from_numpy(candidate_tails)
        # assert (torch.unique(candidate_tails).size()[0] ==
        #         candidate_tails.size()[0])
        # once the corruption entities have been sampled
        # get the head and relation of the triple
        hr = triple[0:2]
        # repeat it k times and shape appropriately
        hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
            .repeat(1, k, 1).transpose(2, 1).squeeze()
        # combine with the corrupted tails
        combined_t = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                               dim=0)
        combined_t = combined_t.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete_t = torch.cat((triples[i, :].unsqueeze(0), combined_t),
                               dim=0).unsqueeze(0)
        # append to the list
        corrupted_triples.append(complete_t)
        # assert np.array_equal(shuffled[tensor_ind, :], initial)

        #################################
        # Create Head Corruptions ########
        ##################################
        # entities_to_filter = head_filter_dict[(relation, tail)]
        # entities = set(range(max_starting_index))
        # filtered = entities - entities_to_filter
        # candidate_heads_np = np.random.choice(np.array(list(
        #     filtered)), size=k, replace=False).astype(np.int32)
        tensor_ind = random.randint(0, shuffled.shape[0] - 1)
        initial2 = shuffled[tensor_ind, :].copy()
        # assert initial.base is None
        # get a random starting index to get the k entities
        start = random.randint(0, max_starting_index)
        # start = np.random.randint(0, max_starting_index + 1)
        candidate_heads = shuffled[tensor_ind, start:start + k].copy()
        # assert candidate_heads.base is None
        # find if the true tail was sampled or any of the tails that need
        # to be filtered
        entities_to_filter = head_filter_dict[(relation, tail)].copy()
        # in_d = entities_to_filter.copy()
        heads_to_filter = candidate_heads
        zero_indices_list = []
        for entity in entities_to_filter:
            heads_to_filter_entity = heads_to_filter - entity
            zero_indices_list.append(
                (heads_to_filter_entity == 0).nonzero()[0])
            # if (heads_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
            #     # assert entity == int(candidate_heads.item((
            #     #         heads_to_filter_entity == 0).nonzero()[0].item(
            #     #
            #     # ))), str(entity) + " " + str(candidate_heads.item((
            #     #         heads_to_filter_entity == 0).nonzero()[0].item(
            #     #
            #     # )))
        zero_indices = np.concatenate(zero_indices_list, axis=0)
        # replace the entities
        # if the starting index was towards the end of the shuffled
        # tensor start sampling replacement candidates before it else
        # start sampling replacements after it
        if start > max_starting_index // 2:
            remaining = zero_indices.shape[0]
            while remaining > 0:
                idx = random.randint(0, start - 1)
                # idx = np.random.randint(0, start)
                sample_entity = shuffled.item((tensor_ind, idx))
                if sample_entity not in entities_to_filter:
                    candidate_heads[
                        zero_indices[remaining - 1]] = sample_entity
                    entities_to_filter.add(sample_entity)
                    remaining -= 1
        else:
            remaining = zero_indices.shape[0]
            while remaining > 0:
                idx = random.randint(start + k, max_index - 1)
                # idx = np.random.randint(start + k, max_index)
                sample_entity = shuffled.item((tensor_ind, idx))
                if sample_entity not in entities_to_filter:
                    candidate_heads[
                        zero_indices[remaining - 1]] = sample_entity
                    entities_to_filter.add(sample_entity)
                    remaining -= 1
        # assert in_d == head_filter_dict[(relation, tail)]
        candidate_heads = torch.from_numpy(candidate_heads)
        # assert (torch.unique(candidate_heads).size()[0] ==
        #         candidate_heads.size()[0])
        # once the corruption entities have been sampled
        # get the head and relation of the triple
        rt = triple[1:3]
        # repeat it k times and shape appropriately
        rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
            .repeat(1, k, 1).transpose(2, 1).squeeze()
        # combine with the corrupted heads
        combined_h = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                               dim=0)
        combined_h = combined_h.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete_h = torch.cat((triples[i, :].unsqueeze(0), combined_h),
                               dim=0).unsqueeze(0)
        # append to the list
        corrupted_triples.append(complete_h)
        # assert np.array_equal(shuffled[tensor_ind, :], initial2)
    assert np.array_equal(shuffled, shuffled_init)

    return torch.cat(corrupted_triples, dim=0)


def corrupt_entities_set(triples, max_starting_index, k, shuffled,
                     head_filter_dict, tail_filter_dict, max_index):
    # The maximum index that we can start sampling from the shuffled
    # tensor and still get k entities
    max_starting_index = max_index
    corrupted_triples = []
    for i in tqdm(range(triples.size()[0])):
        triple = triples[i, :]
        head = triple[0].item()
        relation = triple[1].item()
        tail = triple[2].item()
        ##################################
        # Make the tail corruptions #####
        ##################################
        entities_to_filter = tail_filter_dict[(head, relation)]
        entities = set(range(max_starting_index))
        filtered = entities - entities_to_filter
        candidate_tails_np = np.random.choice(np.array(list(
            filtered)), size=k, replace=False).astype(np.int32)
        candidate_tails = torch.from_numpy(candidate_tails_np)
        # once the corruption entities have been sampled
        # get the head and relation of the triple
        hr = triple[0:2]
        # repeat it k times and shape appropriately
        hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
            .repeat(1, k, 1).transpose(2, 1).squeeze()
        # combine with the corrupted tails
        combined_t = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                               dim=0)
        combined_t = combined_t.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete_t = torch.cat((triples[i, :].unsqueeze(0), combined_t),
                               dim=0).unsqueeze(0)
        # append to the list
        corrupted_triples.append(complete_t)

        #################################
        # Create Head Corruptions ########
        ##################################
        entities_to_filter = head_filter_dict[(relation, tail)]
        entities = set(range(max_starting_index))
        filtered = entities - entities_to_filter
        candidate_heads_np = np.random.choice(np.array(list(
            filtered)), size=k, replace=False).astype(np.int32)
        candidate_heads = torch.from_numpy(candidate_heads_np)
        # once the corruption entities have been sampled
        # get the head and relation of the triple
        rt = triple[1:3]
        # repeat it k times and shape appropriately
        rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
            .repeat(1, k, 1).transpose(2, 1).squeeze()
        # combine with the corrupted heads
        combined_h = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                               dim=0)
        combined_h = combined_h.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete_h = torch.cat((triples[i, :].unsqueeze(0), combined_h),
                               dim=0).unsqueeze(0)
        # append to the list
        corrupted_triples.append(complete_h)

    return torch.cat(corrupted_triples, dim=0)





def plot_unique_counts(indices, title):
    unique_values, counts = torch.unique(torch.tensor(indices,
                                                      dtype=torch.int32),
                                         return_counts=True)

    sns.lineplot(x=unique_values.numpy(), y=counts.numpy())
    plt.title(title)
    plt.savefig(title + ".png", format='png')
    plt.close()


class CorruptLinkGenerator:

    def __init__(self, head_filter_dict: Dict, tail_filter_dict: Dict,
                 entities: torch.tensor, num_tensor=10):
        """
        The constructor
        Parameters
        ----------
        head_filter_dict : The dictionary of the heads to be filtered
        tail_filter_dict : The dictionary of the tails to be filtered
        entities :
        """
        self.max_index = None
        self.shuffled = None
        self.random_gen = np.random.default_rng()
        self.head_filter_dict = head_filter_dict
        self.tail_filter_dict = tail_filter_dict
        self.entities = np.asarray(entities)
        self.num_shuffled = num_tensor
        self.get_shuffled_indices_tensors()


    def get_shuffled_indices_tensors(self):
        # self.max_index = torch.max(self.entities).item() + 1
        # self.shuffled = torch.zeros((self.num_shuffled, self.max_index),
        #                             dtype=torch.int32)
        # for i in range(self.num_shuffled):
        #     self.shuffled[i, :] = torch.randperm(self.max_index)
        self.max_index = np.max(self.entities).astype(np.int32) + 1
        self.shuffled = np.zeros((self.num_shuffled, self.max_index),
                                    dtype=np.int32)
        for i in range(self.num_shuffled):
            self.shuffled[i, :] = self.random_gen.permutation(
                self.max_index)

    def get_parallel_filtered_corrupted_triples(self, triples, k,
                                                num_workers=None):
        """
        A function that returns k corrupted and filtered head and tail triples
        for each triple of the batch provided
        Parameters
        ----------
        triples :The triples to corrupt
        k : the number of corrupted triples to generate for each triple and
        for each of head and tail (2*k) total

        Returns
        -------
        A torch tensor of shape (2 * num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities
        max_starting_index = self.max_index - k
        num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
        if num_workers > 1:
            slice_size = triples.size()[0] // (num_workers - 1) \
                if (num_workers <= triples.size()[0]) else 1
        else:
            slice_size = triples.size()[0]
        corrupted_triples = Parallel(n_jobs=num_workers, backend="loky") \
            (delayed(corrupt_entities)(triples[i:i + slice_size, :],
                                       max_starting_index, k, self.shuffled,
                                       self.head_filter_dict,
                                       self.tail_filter_dict,
                                       self.max_index
                                       ) for i
             in
             tqdm(range(0,
                        triples.size()[0], slice_size)))

        return torch.cat(corrupted_triples, dim=0)

    def get_filtered_corrupted_triples(self, triples, k):
        """
        A function that returns k corrupted and filtered head and tail triples
        for each triple of the batch provided
        Parameters
        ----------
        triples :The triples to corrupt
        k : the number of corrupted triples to generate for each triple and
        for each of head and tail (2*k) total

        Returns
        -------
        A torch tensor of shape (2 * num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities

        max_starting_index = self.max_index - k
        corrupted_triples = []
        for i in tqdm(range(triples.size()[0])):
            triple = triples[i, :]
            head = triple[0].item()
            relation = triple[1].item()
            tail = triple[2].item()
            ##################################
            # Make the tail corruptions #####
            ##################################
            # entities_to_filter = tail_filter_dict[(head, relation)]
            # entities = set(range(max_starting_index))
            # filtered = entities - entities_to_filter
            # candidate_tails_np = np.random.choice(np.array(list(
            #     filtered)), size=k, replace=False).astype(np.int32)
            tensor_ind = random.randint(0, self.shuffled.shape[0]-1)
            # initial = self.shuffled[tensor_ind, :].copy()
            #check if initial is a copy
            # assert initial.base is None
            # get a random starting index to get the k entities
            # start = np.random.randint(low=0, high=max_starting_index + 1)
            start = random.randint(0, max_starting_index)
            candidate_tails = self.shuffled[tensor_ind, start:start + k].copy()
            # assert candidate_tails.base is None
            # find if the true tail was sampled or any of the tails that need
            # to be filtered
            entities_to_filter = self.tail_filter_dict[(head, relation)].copy()
            # in_d = entities_to_filter.copy()
            tails_to_filter = candidate_tails
            zero_indices_list = []
            for entity in entities_to_filter:
                tails_to_filter_entity = tails_to_filter - entity
                zero_indices_list.append(
                    (tails_to_filter_entity == 0).nonzero()[0])
                # if (tails_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
                #     assert entity == candidate_tails.item((
                #             tails_to_filter_entity == 0).nonzero()[0].item())
            zero_indices = np.concatenate(zero_indices_list, axis=0)
            # replace the entities
            # if the starting index was towards the end of the shuffled
            # tensor start sampling replacement candidates before it else
            # start sampling replacements after it
            if start > max_starting_index // 2:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(0, start - 1)
                    # idx = np.random.randint(0, start)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_tails[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            else:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(start + k, self.max_index - 1)
                    # idx = np.random.randint(start + k, max_index)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_tails[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            # assert in_d == self.tail_filter_dict[(head, relation)]
            candidate_tails = torch.from_numpy(candidate_tails)
            # assert (torch.unique(candidate_tails).size()[0] ==
            #         candidate_tails.size()[0])
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            hr = triple[0:2]
            # repeat it k times and shape appropriately
            hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted tails
            combined_t = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                                   dim=0)
            combined_t = combined_t.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_t = torch.cat((triples[i, :].unsqueeze(0), combined_t),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_t)
            # assert np.array_equal(self.shuffled[tensor_ind,:], initial)

            #################################
            # Create Head Corruptions ########
            ##################################
            # entities_to_filter = head_filter_dict[(relation, tail)]
            # entities = set(range(max_starting_index))
            # filtered = entities - entities_to_filter
            # candidate_heads_np = np.random.choice(np.array(list(
            #     filtered)), size=k, replace=False).astype(np.int32)
            tensor_ind = random.randint(0, self.shuffled.shape[0] - 1)
            initial2 = self.shuffled[tensor_ind, :].copy()
            # assert initial.base is None
            # get a random starting index to get the k entities
            start = random.randint(0, max_starting_index)
            # start = np.random.randint(0, max_starting_index + 1)
            candidate_heads = self.shuffled[tensor_ind, start:start + k].copy()
            # assert candidate_heads.base is None
            # find if the true tail was sampled or any of the tails that need
            # to be filtered
            entities_to_filter = self.head_filter_dict[(relation, tail)].copy()
            # in_d = entities_to_filter.copy()
            heads_to_filter = candidate_heads
            zero_indices_list = []
            for entity in entities_to_filter:
                heads_to_filter_entity = heads_to_filter - entity
                zero_indices_list.append(
                    (heads_to_filter_entity == 0).nonzero()[0])
                # if (heads_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
                #     assert entity == int(candidate_heads.item((
                #             heads_to_filter_entity == 0).nonzero()[0].item(
                #
                #     ))), str(entity) + " "+str(candidate_heads.item((
                #             heads_to_filter_entity == 0).nonzero()[0].item(
                #
                #     )))
            zero_indices = np.concatenate(zero_indices_list, axis=0)
            # replace the entities
            # if the starting index was towards the end of the shuffled
            # tensor start sampling replacement candidates before it else
            # start sampling replacements after it
            if start > max_starting_index // 2:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(0, start - 1)
                    # idx = np.random.randint(0, start)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_heads[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            else:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(start + k, self.max_index - 1)
                    # idx = np.random.randint(start + k, max_index)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_heads[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            # assert in_d == self.head_filter_dict[(relation, tail)]
            candidate_heads = torch.from_numpy(candidate_heads)
            # assert (torch.unique(candidate_heads).size()[0] ==
            #         candidate_heads.size()[0])
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            rt = triple[1:3]
            # repeat it k times and shape appropriately
            rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted heads
            combined_h = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                                   dim=0)
            combined_h = combined_h.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_h = torch.cat((triples[i, :].unsqueeze(0), combined_h),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_h)
            # assert np.array_equal(self.shuffled[tensor_ind, :], initial2)

        return torch.cat(corrupted_triples, dim=0)



class CorruptLinkGeneratorEval:

    def __init__(self, head_filter_dict: Dict, tail_filter_dict: Dict,
                 entities: torch.tensor, num_tensor=10):
        """
        The constructor
        Parameters
        ----------
        head_filter_dict : The dictionary of the heads to be filtered
        tail_filter_dict : The dictionary of the tails to be filtered
        entities :
        """
        self.max_index = None
        self.shuffled = None
        self.random_gen = np.random.default_rng()
        self.head_filter_dict = head_filter_dict
        self.tail_filter_dict = tail_filter_dict
        self.entities = np.asarray(entities)
        self.num_shuffled = num_tensor
        self.get_shuffled_indices_tensors()

    def get_shuffled_indices_tensors(self):
        self.max_index = np.max(self.entities).astype(np.int32) + 1
        self.shuffled = np.zeros((self.num_shuffled, self.max_index),
                                    dtype=np.int32)
        for i in range(self.num_shuffled):
            self.shuffled[i, :] = self.random_gen.permutation(
                self.max_index)
    def get_parallel_filtered_corrupted_triples(self, triples, k,
                                                num_workers=None):
        """
        A function that returns k corrupted and filtered head and tail triples
        for each triple of the batch provided
        Parameters
        ----------
        triples :The triples to corrupt
        k : the number of corrupted triples to generate for each triple and
        for each of head and tail (2*k) total

        Returns
        -------
        A torch tensor of shape (2 * num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities
        max_starting_index = self.max_index - k
        num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
        if num_workers > 1:
            slice_size = triples.size()[0] // (num_workers - 1) \
                if (num_workers <= triples.size()[0]) else 1
        else:
            slice_size = triples.size()[0]
        corrupted_triples = Parallel(n_jobs=num_workers, backend="loky") \
            (delayed(corrupt_entities)(triples[i:i + slice_size, :],
                                       max_starting_index, k, self.shuffled,
                                       self.head_filter_dict,
                                       self.tail_filter_dict,
                                       self.max_index
                                       ) for i
             in
             tqdm(range(0,
                        triples.size()[0], slice_size)))

        return torch.cat(corrupted_triples, dim=0)

    def get_filtered_corrupted_triples(self, triples, k):
        """
        A function that returns k corrupted and filtered head and tail triples
        for each triple of the batch provided
        Parameters
        ----------
        triples :The triples to corrupt
        k : the number of corrupted triples to generate for each triple and
        for each of head and tail (2*k) total

        Returns
        -------
        A torch tensor of shape (2 * num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities

        max_starting_index = self.max_index - k
        corrupted_triples = []
        for i in range(triples.size()[0]):
            triple = triples[i, :]
            head = triple[0].item()
            relation = triple[1].item()
            tail = triple[2].item()
            ##################################
            # Make the tail corruptions #####
            ##################################
            # entities_to_filter = tail_filter_dict[(head, relation)]
            # entities = set(range(max_starting_index))
            # filtered = entities - entities_to_filter
            # candidate_tails_np = np.random.choice(np.array(list(
            #     filtered)), size=k, replace=False).astype(np.int32)
            tensor_ind = random.randint(0, self.shuffled.shape[0]-1)
            # initial = self.shuffled[tensor_ind, :].copy()
            #check if initial is a copy
            # assert initial.base is None
            # get a random starting index to get the k entities
            # start = np.random.randint(low=0, high=max_starting_index + 1)
            start = random.randint(0, max_starting_index)
            candidate_tails = self.shuffled[tensor_ind, start:start + k].copy()
            # assert candidate_tails.base is None
            # find if the true tail was sampled or any of the tails that need
            # to be filtered
            entities_to_filter = self.tail_filter_dict[(head, relation)].copy()
            # in_d = entities_to_filter.copy()
            tails_to_filter = candidate_tails
            zero_indices_list = []
            for entity in entities_to_filter:
                tails_to_filter_entity = tails_to_filter - entity
                zero_indices_list.append(
                    (tails_to_filter_entity == 0).nonzero()[0])
                # if (tails_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
                #     assert entity == candidate_tails.item((
                #             tails_to_filter_entity == 0).nonzero()[0].item())
            zero_indices = np.concatenate(zero_indices_list, axis=0)
            # replace the entities
            # if the starting index was towards the end of the shuffled
            # tensor start sampling replacement candidates before it else
            # start sampling replacements after it
            if start > max_starting_index // 2:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(0, start - 1)
                    # idx = np.random.randint(0, start)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_tails[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            else:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(start + k, self.max_index - 1)
                    # idx = np.random.randint(start + k, max_index)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_tails[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            # assert in_d == self.tail_filter_dict[(head, relation)]
            candidate_tails = torch.from_numpy(candidate_tails)
            # assert (torch.unique(candidate_tails).size()[0] ==
            #         candidate_tails.size()[0])
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            hr = triple[0:2]
            # repeat it k times and shape appropriately
            hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted tails
            combined_t = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                                   dim=0)
            combined_t = combined_t.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_t = torch.cat((triples[i, :].unsqueeze(0), combined_t),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_t)
            # assert np.array_equal(self.shuffled[tensor_ind,:], initial)

            #################################
            # Create Head Corruptions ########
            ##################################
            # entities_to_filter = head_filter_dict[(relation, tail)]
            # entities = set(range(max_starting_index))
            # filtered = entities - entities_to_filter
            # candidate_heads_np = np.random.choice(np.array(list(
            #     filtered)), size=k, replace=False).astype(np.int32)
            tensor_ind = random.randint(0, self.shuffled.shape[0] - 1)
            initial2 = self.shuffled[tensor_ind, :].copy()
            # assert initial.base is None
            # get a random starting index to get the k entities
            start = random.randint(0, max_starting_index)
            # start = np.random.randint(0, max_starting_index + 1)
            candidate_heads = self.shuffled[tensor_ind, start:start + k].copy()
            # assert candidate_heads.base is None
            # find if the true tail was sampled or any of the tails that need
            # to be filtered
            entities_to_filter = self.head_filter_dict[(relation, tail)].copy()
            # in_d = entities_to_filter.copy()
            heads_to_filter = candidate_heads
            zero_indices_list = []
            for entity in entities_to_filter:
                heads_to_filter_entity = heads_to_filter - entity
                zero_indices_list.append(
                    (heads_to_filter_entity == 0).nonzero()[0])
                # if (heads_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
                #     assert entity == int(candidate_heads.item((
                #             heads_to_filter_entity == 0).nonzero()[0].item(
                #
                #     ))), str(entity) + " "+str(candidate_heads.item((
                #             heads_to_filter_entity == 0).nonzero()[0].item(
                #
                #     )))
            zero_indices = np.concatenate(zero_indices_list, axis=0)
            # replace the entities
            # if the starting index was towards the end of the shuffled
            # tensor start sampling replacement candidates before it else
            # start sampling replacements after it
            if start > max_starting_index // 2:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(0, start - 1)
                    # idx = np.random.randint(0, start)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_heads[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            else:
                remaining = zero_indices.shape[0]
                while remaining > 0:
                    idx = random.randint(start + k, self.max_index - 1)
                    # idx = np.random.randint(start + k, max_index)
                    sample_entity = self.shuffled.item((tensor_ind, idx))
                    if sample_entity not in entities_to_filter:
                        candidate_heads[
                            zero_indices[remaining - 1]] = sample_entity
                        entities_to_filter.add(sample_entity)
                        remaining -= 1
            # assert in_d == self.head_filter_dict[(relation, tail)]
            candidate_heads = torch.from_numpy(candidate_heads)
            # assert (torch.unique(candidate_heads).size()[0] ==
            #         candidate_heads.size()[0])
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            rt = triple[1:3]
            # repeat it k times and shape appropriately
            rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted heads
            combined_h = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                                   dim=0)
            combined_h = combined_h.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_h = torch.cat((triples[i, :].unsqueeze(0), combined_h),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_h)
            # assert np.array_equal(self.shuffled[tensor_ind, :], initial2)

        return torch.cat(corrupted_triples, dim=0)

    def get_filtered_corrupted_triples_set(self, triples, k):
        """
        A function that returns k corrupted and filtered head and tail triples
        for each triple of the batch provided
        Parameters
        ----------
        triples :The triples to corrupt
        k : the number of corrupted triples to generate for each triple and
        for each of head and tail (2*k) total

        Returns
        -------
        A torch tensor of shape (2 * num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities

        max_starting_index = self.max_index - k
        corrupted_triples = []
        for i in range(triples.size()[0]):
            triple = triples[i, :]
            head = triple[0].item()
            relation = triple[1].item()
            tail = triple[2].item()
            ##################################
            # Make the tail corruptions #####
            ##################################
            entities_to_filter = self.tail_filter_dict[(head, relation)]
            entities = set(range(self.max_index))
            filtered = entities - entities_to_filter
            # candidate_tails_np = np.random.choice(np.array(list(
            #     filtered)), size=k, replace=False).astype(np.int32)
            if len(filtered) >= k:
                candidate_tails_np = np.random.choice(np.array(list(
                    filtered)), size=k, replace=False).astype(np.int32)
            else:
                candidate_tails_np = np.concatenate((np.array(list(
                    filtered)),  np.random.choice(np.array(list(
                    filtered)), size= (k - len(filtered)),
                    replace=False).astype(
                    np.int32)),axis=0)
            # tensor_ind = random.randint(0, self.shuffled.shape[0]-1)
            # initial = self.shuffled[tensor_ind, :].copy()
            # #check if initial is a copy
            # # assert initial.base is None
            # # get a random starting index to get the k entities
            # # start = np.random.randint(low=0, high=max_starting_index + 1)
            # start = random.randint(0, max_starting_index)
            # candidate_tails = self.shuffled[tensor_ind, start:start + k].copy()
            # # assert candidate_tails.base is None
            # # find if the true tail was sampled or any of the tails that need
            # # to be filtered
            # entities_to_filter = self.tail_filter_dict[(head, relation)].copy()
            # # in_d = entities_to_filter.copy()
            # tails_to_filter = candidate_tails
            # zero_indices_list = []
            # for entity in entities_to_filter:
            #     tails_to_filter_entity = tails_to_filter - entity
            #     zero_indices_list.append(
            #         (tails_to_filter_entity == 0).nonzero()[0])
            #     # if (tails_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
            #     #     assert entity == candidate_tails.item((
            #     #             tails_to_filter_entity == 0).nonzero()[0].item())
            # zero_indices = np.concatenate(zero_indices_list, axis=0)
            # # replace the entities
            # # if the starting index was towards the end of the shuffled
            # # tensor start sampling replacement candidates before it else
            # # start sampling replacements after it
            # if start > max_starting_index // 2:
            #     remaining = zero_indices.shape[0]
            #     while remaining > 0:
            #         idx = random.randint(0, start - 1)
            #         # idx = np.random.randint(0, start)
            #         sample_entity = self.shuffled.item((tensor_ind, idx))
            #         if sample_entity not in entities_to_filter:
            #             candidate_tails[
            #                 zero_indices[remaining - 1]] = sample_entity
            #             entities_to_filter.add(sample_entity)
            #             remaining -= 1
            # else:
            #     remaining = zero_indices.shape[0]
            #     while remaining > 0:
            #         idx = random.randint(start + k, self.max_index - 1)
            #         # idx = np.random.randint(start + k, max_index)
            #         sample_entity = self.shuffled.item((tensor_ind, idx))
            #         if sample_entity not in entities_to_filter:
            #             candidate_tails[
            #                 zero_indices[remaining - 1]] = sample_entity
            #             entities_to_filter.add(sample_entity)
            #             remaining -= 1
            # assert in_d == self.tail_filter_dict[(head, relation)]
            candidate_tails = torch.from_numpy(candidate_tails_np)
            # assert (torch.unique(candidate_tails).size()[0] ==
            #         candidate_tails.size()[0])
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            hr = triple[0:2]
            # repeat it k times and shape appropriately
            hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted tails
            combined_t = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                                   dim=0)
            combined_t = combined_t.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_t = torch.cat((triples[i, :].unsqueeze(0), combined_t),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_t)
            # assert np.array_equal(self.shuffled[tensor_ind,:], initial)

            #################################
            # Create Head Corruptions ########
            ##################################
            entities_to_filter = self.head_filter_dict[(relation, tail)]
            entities = set(range(self.max_index))
            filtered = entities - entities_to_filter
            if len(filtered) >= k:
                candidate_heads_np = np.random.choice(np.array(list(
                    filtered)), size=k, replace=False).astype(np.int32)
            else:
                candidate_heads_np = np.concatenate((np.array(list(
                    filtered)),  np.random.choice(np.array(list(
                    filtered)), size=(k - len(filtered)), replace=False).astype(
                    np.int32)),axis=0 )
            # tensor_ind = random.randint(0, self.shuffled.shape[0] - 1)
            # initial2 = self.shuffled[tensor_ind, :].copy()
            # # assert initial.base is None
            # # get a random starting index to get the k entities
            # start = random.randint(0, max_starting_index)
            # # start = np.random.randint(0, max_starting_index + 1)
            # candidate_heads = self.shuffled[tensor_ind, start:start + k].copy()
            # # assert candidate_heads.base is None
            # # find if the true tail was sampled or any of the tails that need
            # # to be filtered
            # entities_to_filter = self.head_filter_dict[(relation, tail)].copy()
            # # in_d = entities_to_filter.copy()
            # heads_to_filter = candidate_heads
            # zero_indices_list = []
            # for entity in entities_to_filter:
            #     heads_to_filter_entity = heads_to_filter - entity
            #     zero_indices_list.append(
            #         (heads_to_filter_entity == 0).nonzero()[0])
            #     # if (heads_to_filter_entity == 0).nonzero()[0].shape[0] > 0:
            #     #     assert entity == int(candidate_heads.item((
            #     #             heads_to_filter_entity == 0).nonzero()[0].item(
            #     #
            #     #     ))), str(entity) + " "+str(candidate_heads.item((
            #     #             heads_to_filter_entity == 0).nonzero()[0].item(
            #     #
            #     #     )))
            # zero_indices = np.concatenate(zero_indices_list, axis=0)
            # # replace the entities
            # # if the starting index was towards the end of the shuffled
            # # tensor start sampling replacement candidates before it else
            # # start sampling replacements after it
            # if start > max_starting_index // 2:
            #     remaining = zero_indices.shape[0]
            #     while remaining > 0:
            #         idx = random.randint(0, start - 1)
            #         # idx = np.random.randint(0, start)
            #         sample_entity = self.shuffled.item((tensor_ind, idx))
            #         if sample_entity not in entities_to_filter:
            #             candidate_heads[
            #                 zero_indices[remaining - 1]] = sample_entity
            #             entities_to_filter.add(sample_entity)
            #             remaining -= 1
            # else:
            #     remaining = zero_indices.shape[0]
            #     while remaining > 0:
            #         idx = random.randint(start + k, self.max_index - 1)
            #         # idx = np.random.randint(start + k, max_index)
            #         sample_entity = self.shuffled.item((tensor_ind, idx))
            #         if sample_entity not in entities_to_filter:
            #             candidate_heads[
            #                 zero_indices[remaining - 1]] = sample_entity
            #             entities_to_filter.add(sample_entity)
            #             remaining -= 1
            # assert in_d == self.head_filter_dict[(relation, tail)]
            candidate_heads = torch.from_numpy(candidate_heads_np)
            # assert (torch.unique(candidate_heads).size()[0] ==
            #         candidate_heads.size()[0])
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            rt = triple[1:3]
            # repeat it k times and shape appropriately
            rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted heads
            combined_h = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                                   dim=0)
            combined_h = combined_h.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_h = torch.cat((triples[i, :].unsqueeze(0), combined_h),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_h)
            # assert np.array_equal(self.shuffled[tensor_ind, :], initial2)

        return torch.cat(corrupted_triples, dim=0)


    def get_filtered_corrupted_triples_and_count(self, triples):
        """
        A function that returns k corrupted and filtered head and tail triples
        for each triple of the batch provided
        Parameters
        ----------
        triples :The triples to corrupt
        k : the number of corrupted triples to generate for each triple and
        for each of head and tail (2*k) total

        Returns
        -------
        A torch tensor of shape (2 * num_triples, k+1, 3)
        """

        # The maximum index that we can start sampling from the shuffled
        # tensor and still get k entities
        max_starting_index = self.max_index
        corrupted_triples = []
        counts = []
        for i in range(triples.size()[0]):
            triple = triples[i, :]
            head = triple[0].item()
            relation = triple[1].item()
            tail = triple[2].item()
            ##################################
            # Make the tail corruptions #####
            ##################################
            # find if the true tail was sampled or any of the tails that need
            # to be filtered
            entities_to_filter = self.tail_filter_dict[(head, relation)]
            entities = set(range(max_starting_index))
            filtered = entities - entities_to_filter
            candidate_tails = torch.Tensor(list(filtered)).int()
            k = candidate_tails.size()[0]
            # k = max_starting_index - len(entities_to_filter) - 1
            # # get a random starting index to get the k entities
            # tensor_id = random.randint(0, self.shuffled.size()[0] - 1)
            # start = 0
            # # random.randint(0, max_starting_index-len(
            # #     entities_to_filter))
            # # start = np.random.randint(low=0, high=max_starting_index + 1)
            # candidate_tails = self.shuffled[tensor_id, start:k+1].clone().detach()
            #
            # tails_to_filter = candidate_tails.clone().detach()
            # zero_indices_list = []
            # for entity in entities_to_filter:
            #     tails_to_filter_entity = tails_to_filter - entity
            #     zero_indices_list.append(
            #         (tails_to_filter_entity == 0).nonzero()[:, 0])
            # zero_indices = torch.cat(zero_indices_list, dim=0)
            # # replace the entities
            # # if the starting index was towards the end of the shuffled
            # # tensor start sampling replacement candidates before it else
            # # start sampling replacements after it
            # remaining = zero_indices.size()[0]
            # rem_entities = self.shuffled[tensor_id, start + k + 1:].clone().detach()
            # idx = 0
            # while remaining > 0:
            #     # idx = random.randint(start + k, self.max_index - 1)
            #     # # idx = np.random.randint(start + k, self.max_index)
            #     sample_entity = rem_entities[idx]
            #     if sample_entity.item() not in entities_to_filter:
            #         candidate_tails[
            #             zero_indices[remaining - 1]] = sample_entity
            #         remaining -= 1
            #     idx += 1
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            hr = triple[0:2]
            # repeat it k times and shape appropriately
            hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted tails
            combined_t = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                                   dim=0)
            combined_t = combined_t.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_t = torch.cat((triples[i, :].unsqueeze(0), combined_t),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_t)
            counts.append(k)

            #################################
            # Create Head Corruptions ########
            ##################################
            # tensor_id = random.randint(0, self.shuffled.size()[0] - 1)
            # get a random starting index to get the k entities
            # start = np.random.randint(0, max_starting_index + 1)
            entities_to_filter = self.head_filter_dict[(relation, tail)]
            entities = set(range(max_starting_index))
            filtered = entities - entities_to_filter
            candidate_heads = torch.Tensor(list(filtered)).int()
            k = candidate_heads.size()[0]
            # k = max_starting_index - len(entities_to_filter) - 1
            # start = 0
            #     #
            #     # random.randint(0, max_starting_index)
            #
            # candidate_heads = self.shuffled[tensor_id, start:k+1]
            # # find if the true tail was sampled or any of the tails that need
            # # to be filtered
            # # entities_to_filter = self.head_filter_dict[(relation, tail)]
            # heads_to_filter = candidate_heads.clone().detach()
            # zero_indices_list = []
            # for entity in entities_to_filter:
            #     heads_to_filter_entity = heads_to_filter - entity
            #     zero_indices_list.append(
            #         (heads_to_filter_entity == 0).nonzero()[:, 0])
            # zero_indices = torch.cat(zero_indices_list, dim=0)
            # # replace the entities
            # # if the starting index was towards the end of the shuffled
            # # tensor start sampling replacement candidates before it else
            # # start sampling replacements after it
            # remaining = zero_indices.size()[0]
            #
            # rem_entities = self.shuffled[tensor_id, start + k + 1:].clone().detach()
            # idx = 0
            # while remaining > 0:
            #     # idx = np.random.randint(start + k, self.max_index)
            #     try:
            #         sample_entity = rem_entities[idx]
            #         if sample_entity.item() not in entities_to_filter:
            #             candidate_heads[
            #                 zero_indices[remaining - 1]] = sample_entity
            #             remaining -= 1
            #         idx += 1
            #     except:
            #         print()
            # once the corruption entities have been sampled
            # get the head and relation of the triple
            rt = triple[1:3]
            # repeat it k times and shape appropriately
            rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
                .repeat(1, k, 1).transpose(2, 1).squeeze()
            # combine with the corrupted heads
            combined_h = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                                   dim=0)
            combined_h = combined_h.transpose(1, 0)
            # add the uncorrupted triple in the beginning
            complete_h = torch.cat((triples[i, :].unsqueeze(0), combined_h),
                                   dim=0).unsqueeze(0)
            # append to the list
            corrupted_triples.append(complete_h)
            counts.append(k)
        return counts, corrupted_triples

class AllCorruptHeadGenerator:

    def __init__(self, filter_dict: Dict, entities: torch.tensor):
        self.max_index = None
        self.shuffled = None
        self.filter_dict = filter_dict
        self.entities = entities
        self.shuffle_entities_tensor()

    def shuffle_entities_tensor(self):
        self.max_index = torch.max(self.entities).item() + 1
        self.entities = torch.randperm(self.max_index)

    def get_corrupted_triple(self, triple):
        """
        A function that returns all the possible corrupted and filtered triples
        for each triple
        Parameters
        ----------
        triple : The triple to corrupt

        Returns
        -------
        A torch tensor of shape (num_triples, 3)
        """

        relation = triple[1].item()
        tail = triple[2].item()
        # fifilter out the actual head and any of the true heads that may exist
        entities_to_filter = self.filter_dict[(relation, tail)]
        heads_to_filter = self.entities
        zero_indices_list = []
        # get the indices that shoud be excluded
        for entity in entities_to_filter:
            heads_to_filter_entity = heads_to_filter - entity
            zero_indices_list.append(
                (heads_to_filter_entity == 0).nonzero()[:, 0])
        zero_indices = torch.cat(zero_indices_list, dim=0)
        # generate a mask with False at the indices to exclude
        mask = torch.ones(self.entities.size()[0], dtype=torch.bool)
        mask[zero_indices] = False
        # select the appropriate heads
        candidate_heads = self.entities[mask]

        rt = triple[1:3]
        # repeat it num heads times and shape appropriately
        rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
            .repeat(1, candidate_heads.size()[0], 1).transpose(2, 1).squeeze()
        # combine with the corrupted heads
        combined = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                             dim=0)
        combined = combined.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete = torch.cat((triple.unsqueeze(0), combined),
                             dim=0).unsqueeze(0)
        return complete


class AllCorruptTailGenerator:

    def __init__(self, filter_dict: Dict, entities: torch.tensor):
        self.max_index = None
        self.shuffled = None
        self.filter_dict = filter_dict
        self.entities = entities
        self.shuffle_entities_tensor()

    def shuffle_entities_tensor(self):
        self.max_index = torch.max(self.entities).item() + 1
        self.entities = torch.randperm(self.max_index)

    def get_corrupted_triple(self, triple):
        """
        A function that returns all the possible corrupted and filtered triples
        for each triple
        Parameters
        ----------
        triple : The triple to corrupt

        Returns
        -------
        A torch tensor of shape (num_triples, 3)
        """

        relation = triple[1].item()
        head = triple[0].item()
        # filter out the actual tail and any of the true tails that may exist
        entities_to_filter = self.filter_dict[(head, relation)]
        tails_to_filter = self.entities
        zero_indices_list = []
        # get the indices that shoud be excluded
        for entity in entities_to_filter:
            tails_to_filter_entity = tails_to_filter - entity
            zero_indices_list.append(
                (tails_to_filter_entity == 0).nonzero()[:, 0])
        zero_indices = torch.cat(zero_indices_list, dim=0)
        # generate a mask with False at the indices to exclude
        mask = torch.ones(self.entities.size()[0], dtype=torch.bool)
        mask[zero_indices] = False
        # select the appropriate tails
        candidate_tails = self.entities[mask]

        hr = triple[0:2]
        # repeat it num heads times and shape appropriately
        hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
            .repeat(1, candidate_tails.size()[0], 1).transpose(2, 1).squeeze()
        # combine with the corrupted heads
        combined = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                             dim=0)
        combined = combined.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete = torch.cat((triple.unsqueeze(0), combined),
                             dim=0).unsqueeze(0)
        return complete


class AllCorruptTailGeneratorUnfiltered:

    def __init__(self, entities: torch.tensor):
        self.max_index = None
        self.shuffled = None
        self.entities = entities
        self.shuffle_entities_tensor()

    def shuffle_entities_tensor(self):
        self.max_index = torch.max(self.entities).item() + 1
        # self.entities = torch.randperm(self.max_index)

    def get_corrupted_triple(self, triple):
        """
        A function that returns all the possible corrupted and filtered triples
        for each triple
        Parameters
        ----------
        triple : The triple to corrupt

        Returns
        -------
        A torch tensor of shape (num_triples, 3)
        """

        relation = triple[1].item()
        head = triple[0].item()
        tail = triple[2].item()
        # filter out the actual tail and any of the true tails that may exist
        tails_to_filter = self.entities
        zero_indices_list = []
        # get the indices that should be excluded
        tails_to_filter_entity = tails_to_filter - tail
        candidate_tails = self.entities[tails_to_filter_entity != 0]
        # zero_indices = (tails_to_filter_entity == 0).nonzero()[:, 0].squeeze()
        # # generate a mask with False at the indices to exclude
        # mask = torch.ones(self.entities.size()[0], dtype=torch.bool)
        # mask[zero_indices] = False
        # # select the appropriate tails
        # candidate_tails = self.entities[mask]

        hr = triple[0:2]
        # repeat it num heads times and shape appropriately
        hr_repeated = hr.unsqueeze(0).unsqueeze(1) \
            .repeat(1, candidate_tails.size()[0], 1).transpose(2, 1).squeeze()
        # combine with the corrupted heads
        combined = torch.cat((hr_repeated, candidate_tails.unsqueeze(0)),
                             dim=0)
        combined = combined.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete = torch.cat((triple.unsqueeze(0), combined),
                             dim=0).unsqueeze(0)
        return complete


class AllCorruptHeadGeneratorUnfiltered:

    def __init__(self,  entities: torch.tensor):
        self.max_index = None
        self.shuffled = None
        self.entities = entities
        self.shuffle_entities_tensor()

    def shuffle_entities_tensor(self):
        self.max_index = torch.max(self.entities).item() + 1
        # self.entities = torch.randperm(self.max_index)

    def get_corrupted_triple(self, triple):
        """
        A function that returns all the possible corrupted and unfiltered
        triples for each triple
        Parameters
        ----------
        triple : The triple to corrupt

        Returns
        -------
        A torch tensor of shape (num_triples, 3)
        """

        relation = triple[1].item()
        head = triple[0].item()
        tail = triple[2].item()
        # filter out the actual tail and any of the true tails that may exist
        heads_to_filter = self.entities
        zero_indices_list = []
        # get the indices that should be excluded (just the head)
        heads_to_filter_entity = heads_to_filter - head
        candidate_heads = self.entities[heads_to_filter_entity != 0]
        # zero_indices = (heads_to_filter_entity == 0).nonzero()[:, 0].squeeze()
        # # generate a mask with False at the indices to exclude
        # mask = torch.ones(self.entities.size()[0], dtype=torch.bool)
        # mask[zero_indices] = False
        # # select the appropriate tails
        # candidate_heads = self.entities[mask]

        rt = triple[1:3]
        # repeat it num heads times and shape appropriately
        rt_repeated = rt.unsqueeze(0).unsqueeze(1) \
            .repeat(1, candidate_heads.size()[0], 1).transpose(2, 1).squeeze()
        # combine with the corrupted heads
        combined = torch.cat((candidate_heads.unsqueeze(0), rt_repeated),
                             dim=0)
        combined = combined.transpose(1, 0)
        # add the uncorrupted triple in the beginning
        complete = torch.cat((triple.unsqueeze(0), combined),
                             dim=0).unsqueeze(0)
        return complete


def generate_negative_triples(triplestore, num_negatives, triple_corruptor,
                              parallel, num_workers=None):
    """
    Generates a batched tensor of interleaved positive and (the corresping)
    negative triples using the given corruptor.

    Parameters
    ----------
    triplestore : torch.tensor
        A tensor of shape (num_triples, 3) holding the positive triples.
    num_negatives : int
        The number of negatives per positive triple to generate.
    triple_corruptor : TripleCorruptor
        A triple corruptor, generating negatives based on a strategy.

    """
    num_positives = len(triplestore)  # triplestore with all positives
    if num_negatives <= 0:  # sanity check on negatives
        raise ValueError(f"Cannot generate {num_negatives} negatives")
    logger.info(f"Creating {num_negatives} negs per {num_positives}")
    if not parallel:
        corruptions = triple_corruptor.get_filtered_corrupted_triples(
            triplestore, num_negatives)  # generate full corruption tensor
    else:
        corruptions = triple_corruptor.get_parallel_filtered_corrupted_triples(
            triplestore, num_negatives, num_workers)  # generate full corruption tensor
    positive_triples = corruptions[:, 0, :]
    negative_triples = corruptions[:, 1:, :]
    negative_triples = negative_triples.reshape(-1, 3)
    assert len(negative_triples) == num_positives * num_negatives \
           or len(negative_triples) == num_positives * num_negatives * 2
    assert torch.all(positive_triples.unique(dim=0) == triplestore)

    return torch.concat([positive_triples, negative_triples])

def generate_negative_batches(triplestore, num_negatives, triple_corruptor,
                              parallel, num_workers=None):
    """
    Generates a batched tensor of interleaved positive and (the corresping)
    negative triples using the given corruptor.

    Parameters
    ----------
    triplestore : torch.tensor
        A tensor of shape (num_triples, 3) holding the positive triples.
    num_negatives : int
        The number of negatives per positive triple to generate.
    triple_corruptor : TripleCorruptor
        A triple corruptor, generating negatives based on a strategy.

    """
    num_positives = len(triplestore)  # triplestore with all positives
    if num_negatives <= 0:  # sanity check on negatives
        raise ValueError(f"Cannot generate {num_negatives} negatives")
    logger.info(f"Creating {num_negatives} negs per {num_positives}")
    if not parallel:
        corruptions = triple_corruptor.get_filtered_corrupted_triples(
            triplestore, num_negatives)  # generate full corruption tensor
    else:
        corruptions = triple_corruptor.get_parallel_filtered_corrupted_triples(
            triplestore, num_negatives, num_workers)  # generate full corruption tensor
    positive_triples = corruptions[:, 0, :]
    negative_triples = corruptions[:, 1:, :]
    negative_triples = negative_triples.reshape(-1, 3)
    assert len(negative_triples) == num_positives * num_negatives \
           or len(negative_triples) == num_positives * num_negatives * 2
    assert torch.all(positive_triples.unique(dim=0) == triplestore)

    return corruptions