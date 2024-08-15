"""
Common utilities for manipulating ER paths.

Note: some of this methods are refactoring the ``UnrolledPathDataset''.

"""
import logging
from tqdm import tqdm
from typing import List, Dict

import torch
import pandas as pd

import data_utils as du

logger = logging.getLogger(__name__)


class InconsistentVocabulary(Exception):
    pass  # raised when the input vocab is not consistent with the data


def check_relation_vocabulary(paths: List[List[int]], vocabulary: List):
    """
    Check whether a vocabulary subsumes the one implied by a list of paths.
    
    Parameters
    ----------
    paths : List[List[int]]
        A list containing relational paths, encoded as a list of integers.
    vocabulary : List[int]
        A vocabulary of tokens (including special tokens) to sanity check.

    """
    for path in tqdm(paths, desc="Vocabulary check"):
        if not all([r in vocabulary for r in path]):
            return False
    # Vocabulary is good at this stage
    return True


def create_path_indexing(entity_paths: torch.tensor):
    """
    Create a dictionary indexing paths based on the first and last entities.

    Parameters
    ----------
    entity_paths : torch.tensor
        A 2D tensor where each record correponds to an entity path.

    Returns
    -------
    path_index : dict
        Indexes the incoming and the outgoing paths given an entity, if present.
        For example, ``path_index['in'][123]`` will return a list of indexes
        that are relative to ``entity_paths`` where each of them corresponds to
        an incoming path of the entity ``123``. Analogous for outgoing paths.

    """

    path_index = {"in": {}, "out": {}}
    for i in tqdm(range(len(entity_paths)), desc="H-T path indexing"):
        # Retrieving start and end node from entity paths
        start, end = entity_paths[i][[0, -1]]
        start, end = start.item(), end.item()
        # Populating path index for incoming paths
        if end not in path_index["in"]:
            path_index["in"][end] = [i]
        else:  # node already in dict
            path_index["in"][end].append(i)
        # Populating path index for outgoing paths
        if start not in path_index["out"]:
            path_index["out"][start] = [i]
        else:  # node already in dict
            path_index["out"][start].append(i)

    return path_index


def encode_relcontext_freqs(relcontext, num_entities=None,
                            num_relations=None, offset:int=0):
    """

    Parameters
    ----------
    relcontext : str or pd.DataFrame
        A path to a CSV or DataFrame containing the relational context per node.
    num_entities : int, optional
        The number of entiites for which the relational context vectors will be
        created; as this is necessary to initialiase data structures.
    num_relations : int, optional
        The number of unique relations in the vocabulary.

    Returns
    -------
    entity_rcs : torch.tensor
        A matrix indexed by entities where columns are relational context freqs
    """
    # Creating and initialising data structures if information is not provided
    if isinstance(relcontext, str):
        relcontext = pd.read_csv(relcontext, sep=',')
        du.listify_columns(relcontext, "edges")
    if num_relations is None or num_entities is None:
        raise NotImplementedError
    # offsetting num relations to include mask and padding tokens
    num_relations = num_relations + offset
    num_entities = num_entities + offset
    # Allocating a matrix when rows denote entities, and columns rels in-out
    entity_emb = torch.zeros(
        (num_entities, 2 * num_relations), dtype=torch.int32)
    entity_emb_inc = torch.zeros((num_entities, num_relations),
                                 dtype=torch.int32)
    entity_emb_out = torch.zeros((num_entities, num_relations),
                                 dtype=torch.int32)
    # Populating the matrix starting from the
    for _, row in relcontext.iterrows():
        entity = int(row['node_id']) + offset
        context = torch.IntTensor(row['edges'])
        indices, counts = torch.unique(context, return_counts=True)
        # offseting indices to skip over mask and padding
        indices = indices + offset
        if row['direction'] == 'in':
            entity_emb[entity, indices] = counts.type(torch.int32)
            entity_emb_inc[entity, indices] = counts.type(torch.int32)
        else:
            entity_emb[entity, indices + num_relations] = counts.type(
                torch.int32)
            entity_emb_out[entity, indices] = counts.type(torch.int32)
    return (entity_emb.type(torch.float32),
            entity_emb_inc.type(torch.float32),
            entity_emb_out.type(torch.float32))
    # return entity_emb


def create_protopaths(head, tail, path_index, triples):
    """
    Create a list of protopaths by combining incoming and outgoing paths with
    respect to the given head and tail entities. Incoming and outgoing triples
    are used whenever a corresponding path is not found from the pathindex.
    A protopath comes in the form: (in_idx, in_type, out_idx, out_type).
    """
    def clean_triple_context(context):
        # Make sure single contexts are wrapped in a list and defaulted to None
        context = [context] if isinstance(context, int) else context
        return [None] if len(context) == 0 else context

    in_paths, in_triples = [], []
    out_paths, out_triples = [], []

    # Get incoming paths or triples from head
    if head.item() in path_index["in"]:
        in_paths = path_index["in"][head.item()]
    in_triples = (triples[:, 2] == head).nonzero().squeeze().tolist()
    # Get outgoing paths or triples from tail
    if tail.item() in path_index["out"]:
        out_paths = path_index["out"][tail.item()]
    out_triples = (triples[:, 0] == tail).nonzero().squeeze().tolist()

    in_context, in_type = (in_paths, "path") \
        if len(in_paths) > 0 else (in_triples, "triple")
    out_context, out_type = (out_paths, "path") \
        if len(out_paths) > 0 else (out_triples, "triple")
    # Clean triple contexts before combinations are performed
    in_context = clean_triple_context(in_context)
    out_context = clean_triple_context(out_context)
    protopaths = [(inc, in_type, outc, out_type)
                    for inc in in_context for outc in out_context]

    return protopaths


def create_contextpaths(entity, path_index, triples):
    """
    Given an entity, this function grabs incoming and outgoing paths. When paths
    are not found, triples are inserted as len-1 paths. Note that full paths are
    not returned, but only indexes to access them in path_stores and `triples`.
    """
    def index_context(ctype):
        to_list = lambda x: [x] if not isinstance(x, list) else x
        i = 2 if ctype == "in" else 0
        if entity in path_index[ctype]:  # choose paths, if available
            icontext = [(idx, "path") for idx in path_index[ctype][entity]]
        else: # retrieve all incoming triples if no path is found
            icontext = [(idx, "triple") for idx in to_list((triples[:,
                                                        i] == entity)\
                        .nonzero().squeeze().tolist())]
        return icontext

    if torch.is_tensor(entity):
        entity = entity.item()

    in_paths, out_paths = index_context("in"), index_context("out")
    return in_paths, out_paths


def get_entfocused_positionals(center_idx, path_len):
    """
    Creates an incremental positional encoding where the entity to focus (the
    centre of the path) receives 0 and increments where moving away from it. 

    Parameters
    ----------
    center_idx : int
        The index of the entity that is used to contextualise the path.
    path_len : int
        The length of the entity-relation path in which the entity lies.

    """
    return torch.cat([torch.arange(0, center_idx+1).flip(0),
                      torch.arange(center_idx, path_len-1) - (center_idx-1)])

