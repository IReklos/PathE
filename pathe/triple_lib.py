"""
The triple library (triple_lib) provides utilities for manipulating triples.

"""
import os
from collections import defaultdict, Counter

import pandas as pd
import torch

from data_utils import listify_columns


def get_class_weights(triples: torch.tensor, tokens_to_idxs):
    """
    A function that calculates class weights for unbalanced datasets.

    Parameters
    ----------
    triples : torch.tensor
        The triples to consider for weight computation.
    tokens_to_idxs : dict
        The dict mapping ids tod token_ids used in training.
    """
    relations = triples[:, 1]
    unique, counts = torch.unique(relations, return_counts=True, sorted=True)
    weights = torch.ones(len(tokens_to_idxs), dtype=torch.float32)
    # weight is inverse of frequency
    # rel_weights = 1 / counts.float()
    # weight is inverse of ratio
    # rel_weights = 1/(counts.float()/torch.sum(counts))
    # weight is calculated according to tensorflow tutorial
    # https://ai.stackexchange.com/questions/20249/how-are-weights
    # -for-weighted-x-entropy-loss-on-imbalanced-data-calculated
    rel_weights = (1 / counts.float()) * (torch.sum(counts) / counts.size()[0])
    for i in range(rel_weights.size()[0]):
        index = tokens_to_idxs[i]
        weights[index] = rel_weights[i]
    weights[tokens_to_idxs['PAD']] = 0.0
    weights[tokens_to_idxs['MSK']] = 0.0
    return weights

def get_class_weights_without_special_tokens(triples: torch.tensor):
    """
    A function that calculates class weights for unbalanced datasets.

    Parameters
    ----------
    triples : torch.tensor
        The triples to consider for weight computation.
    """
    relations = triples[:, 1]
    unique, counts = torch.unique(relations, return_counts=True, sorted=True)
    weights = torch.ones((unique.size()[0]), dtype= torch.float32)
    # weight is inverse of frequency
    # rel_weights = 1 / counts.float()
    # weight is inverse of ratio
    # rel_weights = 1/(counts.float()/torch.sum(counts))
    # weight is calculated according to tensorflow tutorial
    # https://ai.stackexchange.com/questions/20249/how-are-weights
    # -for-weighted-x-entropy-loss-on-imbalanced-data-calculated
    rel_weights = (1 / counts.float()) * (torch.sum(counts) / counts.size()[0])
    return rel_weights


def get_dataset_dir(dataset: str):
    dataset_dir = os.path.join(os.path.join(
        os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                     "data"), "path_datasets"), dataset)
    return dataset_dir

def get_relational_context_encoding_matrix(relational_con: pd.DataFrame,
                                           num_nodes: int, num_rel: int,
                                           offset: int=2):
    """
    returns the relational context encodings of nodes
    Parameters
    ----------
    relational_con : The pandas Dataframe with columns ['node_id',
    'direction','edges']
    num_nodes : The number of unique nodes in the KG
    num_rel : The number of unique relations in the KG

    Returns
    -------

    """
    if offset != 0:
        num_nodes = num_nodes + offset
        num_rel = num_rel + offset
    relational_con['edges'] = relational_con['edges'].apply(
        lambda x: [int(i) for i in x.split()])

    entity_emb= torch.zeros((num_nodes, 2 * num_rel),
                                  dtype=torch.int32)
    entity_emb_inc = torch.zeros((num_nodes, num_rel),
                                 dtype=torch.int32)
    entity_emb_out = torch.zeros((num_nodes, num_rel),
                                 dtype=torch.int32)
    for index, row in relational_con.iterrows():
        entity = int(row['node_id']) + offset
        con_type = row['direction']
        context = torch.IntTensor(row['edges'])
        indices, counts = torch.unique(context, return_counts=True)
        indices = indices + offset
        if con_type == 'in':
            entity_emb[entity, indices] = counts.type(torch.int32)
            entity_emb_inc[entity, indices] = counts.type(torch.int32)
        else:
            entity_emb[entity, indices + num_rel] = counts.type(
                torch.int32)
            entity_emb_out[entity, indices] = counts.type(torch.int32)
    return (entity_emb.type(torch.float32),
            entity_emb_inc.type(torch.float32),
            entity_emb_out.type(torch.float32))


def get_adjacency_matrix(triples: torch.tensor):
    """
    Get the adjacency matrix of a KG based on the triples
    Parameters
    ----------
    triples : A troch tensor of the triples with shape (num triples, 3)

    Returns
    -------

    The adjacency matrix of shape (num triples, num triples)

    """
    unique_nodes = torch.unique(torch.cat((triples[:, 0],
                                           triples[:, 2])))
    adj_matrix = torch.zeros((unique_nodes.size()[0], unique_nodes.size()[0]),
                             dtype=torch.int32)
    adj_matrix[triples[:, 0], triples[:, 2]] = 1
    return adj_matrix


def get_triple_completions(triples: torch.tensor, triple: torch.tensor,
                           missing: int):
    """
    A function that finds incoming and outgoing triples in the triples
    dataset and adds them to the triple to create short paths
    Parameters
    ----------
    missing : The part where paths are missing in int format (0 for head,
    1 for tail, 2 for both)
    triples : The triples dataset (num triples, 3)
    triple : The triple (3)

    Returns
    -------

    """
    # get the triples that end at the triple head and the triples that start
    # from the triple tail
    incoming = triples[(triples[:, 2] == triple[0]).nonzero()[:, 0], :]
    outgoing = triples[(triples[:, 0] == triple[2]).nonzero()[:, 0], :]

    # repeat the triple to concatenate and create paths
    triple_repeated_incoming = triple.repeat(incoming.size()[0], 1)
    triple_repeated_outgoing = triple.repeat(outgoing.size()[0], 1)

    # depending on if incoming or outgoing (or both) triples were found,
    # concatenate and return the appropriately shaped tensor
    if incoming.size()[0] > 0:
        incoming_paths = torch.cat((incoming, triple_repeated_incoming[:, 1:3]),
                                   dim=-1)
        # print(incoming_paths)
    if outgoing.size()[0] > 0:
        outgoing_paths = torch.cat((triple_repeated_outgoing[:, 0:2], outgoing),
                                   dim=-1)
        # print(outgoing_paths)
    if incoming.size()[0] > 0 and outgoing.size()[0] > 0:
        final = torch.cat((incoming_paths, outgoing_paths), dim=0)
    elif incoming.size()[0] > 0:
        final = incoming_paths
    elif outgoing.size()[0] > 0:
        final = outgoing_paths
    else:
        final = triple
    # print(final)
    return final


def get_missing_end_triples(triples: torch.tensor, triple: torch.tensor,
                            missing: int):
    """
    A function that finds incoming and outgoing triples in the triples
    dataset and adds them to the triple to create short paths
    Parameters
    ----------
    missing : The part where paths are missing in int format (0 for head,
    1 for tail, 2 for both)
    triples : The triples dataset (num triples, 3)
    triple : The triple (3)

    Returns
    -------
    A tuple with the requested triples (if they exist)
    If requesting only incoming to head (0)  a tuple of (incoming, None,
    None)
    If requesting only outgoing from tail (1) a tuple of (None, outgoing, None)
    If requesting both, a tuple of (incoming, outgoing, all combinations)
    If none could be found a tuple of (None, None, None) is returned
    """
    # get the triples that end at the triple head and the triples that start
    # from the triple tail depending on what is missing
    if missing == 0:
        incoming = triples[(triples[:, 2] == triple[0]).nonzero()[:, 0], :]
        if incoming.size()[0] > 0:
            return incoming, None, None
        else:
            return None, None, None
    elif missing == 1:
        outgoing = triples[(triples[:, 0] == triple[2]).nonzero()[:, 0], :]
        if outgoing.size()[0] > 0:
            return None, outgoing, None
        else:
            return None, None, None
    else:
        # depending on if incoming or outgoing (or both) triples were found,
        # concatenate and return the appropriately shaped tensors of
        # incoming, outgoing and all their combinations
        incoming = triples[(triples[:, 2] == triple[0]).nonzero()[:, 0], :]
        outgoing = triples[(triples[:, 0] == triple[2]).nonzero()[:, 0], :]
        if incoming.size()[0] > 0 and outgoing.size()[0] > 0:
            triples_r = torch.repeat_interleave(incoming, outgoing.size()[0],
                                                dim=0)
            t_out_r = outgoing.repeat(incoming.size()[0], 1)
            perm = torch.cat((triples_r, t_out_r), dim=-1)
            return incoming, outgoing, perm
        elif incoming.size()[0] > 0:
            return incoming, None, None
        elif outgoing.size()[0] > 0:
            return None, outgoing, None
        else:
            return None, None, None


def make_relation_filter_dict(train_triples: torch.tensor,
                              val_triples: torch.tensor,
                              test_triples: torch.tensor,
                              tokens_to_idxs):
    """
    Creates a dictionary which maps each head and tail to a set of the
    ground truth relations between them. To be used for filtering during
    metric calculation.
    """
    filter_dict = defaultdict(set)
    all_triples = torch.cat((train_triples,
                             val_triples,
                             test_triples), dim=0)
    for i in range(all_triples.size()[0]):
        filter_dict[(all_triples[i, 0].item(),
                     all_triples[i, 2].item())].add(
            tokens_to_idxs[all_triples[i, 1].item()])
    return filter_dict


def make_relation_filter_dict_no_sp_tokens(train_triples: torch.tensor,
                                           val_triples: torch.tensor,
                                           test_triples: torch.tensor):
    filter_dict = defaultdict(set)
    all_triples = torch.cat((train_triples,
                             val_triples,
                             test_triples), dim=0)
    for i in range(all_triples.size()[0]):
        filter_dict[(all_triples[i, 0].item(),
                     all_triples[i, 2].item())].add(
            all_triples[i, 1].item())
    return filter_dict


def make_head_tail_dicts(train_triples: torch.tensor,
                         val_triples: torch.tensor,
                         test_triples: torch.tensor):
    """
    Creates a dictionary which maps each head and relation to a set of the
    ground truth tails between them and a dictionary which maps each relation
    and tail to a set of the ground truth heads between them. To be used for
    filtering during metric calculation.

    Parameters
    ----------
    train_triples : the train triples in torch.tensor format [h,r,t]
    val_triples : the validation triples in torch.tensor format [h,r,t]
    test_triples : the test triples in torch.tensor format [h,r,t]

    """
    tail_filter_dict = defaultdict(set)
    head_filter_dict = defaultdict(set)
    all_triples = torch.cat((train_triples,
                             val_triples,
                             test_triples), dim=0)
    for i in range(all_triples.size()[0]):
        head_filter_dict[(all_triples[i, 1].item(),
                          all_triples[i, 2].item())].add(
            all_triples[i, 0].item())
        tail_filter_dict[(all_triples[i, 0].item(),
                          all_triples[i, 1].item())].add(
            all_triples[i, 2].item())
    return head_filter_dict, tail_filter_dict


def get_unique_entities(train_triples: torch.tensor,
                        val_triples: torch.tensor,
                        test_triples: torch.tensor):
    """
    Returns all the unique entities in the KG sorted in ascending order.

    Parameters
    ----------
    train_triples : the train triples in torch.tensor format [h,r,t]
    val_triples : the validation triples in torch.tensor format [h,r,t]
    test_triples : the test triples in torch.tensor format [h,r,t]

    """
    all_triples = torch.cat((train_triples,
                             val_triples,
                             test_triples), dim=0)
    unique_entities = torch.stack((all_triples[:, 0], all_triples[:, 2]),
                                  dim=0).unique(sorted=True)
    return unique_entities


def get_dead_ends(relcontext_path):
    """
    Returns the entities pointing to another entity with no further outgoing
    connection, aka dead ends. The function also prints statistics.

    Parameters
    ----------
    relcontext_path : str
        Path to the relational context CSV.
    """
    relcontext_df = pd.read_csv(relcontext_path)
    listify_columns(relcontext_df, "edges")  # same for rel context
    relcontext_dict = {i: {"in": [], "out": []}
                    for i in list(set(relcontext_df["node_id"]))}

    for _, record in relcontext_df.iterrows():
        relcontext_dict[record["node_id"]][record["direction"]] = \
            (len(record["edges"]), list(set(record["edges"])))

    dead_ends = [(idx, rels["out"][1]) for idx, rels in relcontext_dict.items()
                if rels["in"] == [] and rels["out"][0] == 1]
    dead_ends_no = len(dead_ends)  # no. of dead end entities

    print(f"Dead ends: {dead_ends_no} "
        f"({(dead_ends_no / len(relcontext_dict) * 100):.2f}%)")
    dedrel_cnt = Counter([b[0] for a, b in dead_ends])
    for face, count in dedrel_cnt.items():
        print(f"(*) {face}: {count} ({count/dead_ends_no:.2%})")

    return dead_ends
