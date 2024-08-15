import os
import torch


def count_unique_nodes(triples: torch.Tensor):
    """
    A function that combines two torch.Tensors and calculates the number of unique values
    :param triples: A torch.Tensor where each row is the (head,relation,tail)
    :return: The number of unique elements
    """
    heads = triples[:, 0]
    tails = triples[:, 2]
    num_unique = torch.unique(torch.stack((heads, tails), dim=1)).shape[0]
    return num_unique


def count_unique_edges(triples: torch.Tensor):
    """
    A function that calculates the number of unique edges
    :param triples: A torch.Tensor where each row is the (head,relation,tail)
    :return: The number of unique edges
    """
    num_unique = torch.unique(triples[:, 1]).shape[0]
    return num_unique



def check_and_make_dir(directory: str):
    """
    Checks if a directory exists and if not creates it
    :param directory: The path to the directory
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


# static values for special tokens
NOTHING_TOKEN = -99
CLS_TOKEN = -1
MASK_TOKEN = -10
PADDING_TOKEN = -100
SEP_TOKEN = -2

METRIC_DICT = {'h_pessimist_mrr': 'head.pessimistic.mean_reciprocal_rank',
               'h_realist_mrr': 'head.realistic.mean_reciprocal_rank',
               'h_optimist_mrr': 'head.optimistic.mean_reciprocal_rank',
               't_pessimist_mrr': 'tail.pessimistic.mean_reciprocal_rank',
               't_realist_mrr': 'tail.realistic.mean_reciprocal_rank',
               't_optimist_mrr': 'tail.optimistic.mean_reciprocal_rank',
               'b_pessimist_mrr': 'both.pessimistic.mean_reciprocal_rank',
               'b_realist_mrr': 'both.realistic.mean_reciprocal_rank',
               'b_optimist_mrr': 'both.optimistic.mean_reciprocal_rank',
               'h_pessimist_hits_1': 'head.pessimistic.hits_at_1',
               'h_realist_hits_1': 'head.realistic.hits_at_1',
               'h_optimist_hits_1': 'head.optimistic.hits_at_1',
               't_pessimist_hits_1': 'tail.pessimistic.hits_at_1',
               't_realist_hits_1': 'tail.realistic.hits_at_1',
               't_optimist_hits_1': 'tail.optimistic.hits_at_1',
               'b_pessimist_hits_1': 'both.pessimistic.hits_at_1',
               'b_realist_hits_1': 'both.realistic.hits_at_1',
               'b_optimist_hits_1': 'both.optimistic.hits_at_1',
               'h_pessimist_hits_3': 'head.pessimistic.hits_at_3',
               'h_realist_hits_3': 'head.realistic.hits_at_3',
               'h_optimist_hits_3': 'head.optimistic.hits_at_3',
               't_pessimist_hits_3': 'tail.pessimistic.hits_at_3',
               't_realist_hits_3': 'tail.realistic.hits_at_3',
               't_optimist_hits_3': 'tail.optimistic.hits_at_3',
               'b_pessimist_hits_3': 'both.pessimistic.hits_at_3',
               'b_realist_hits_3': 'both.realistic.hits_at_3',
               'b_optimist_hits_3': 'both.optimistic.hits_at_3',
               'h_pessimist_hits_5': 'head.pessimistic.hits_at_5',
               'h_realist_hits_5': 'head.realistic.hits_at_5',
               'h_optimist_hits_5': 'head.optimistic.hits_at_5',
               't_pessimist_hits_5': 'tail.pessimistic.hits_at_5',
               't_realist_hits_5': 'tail.realistic.hits_at_5',
               't_optimist_hits_5': 'tail.optimistic.hits_at_5',
               'b_pessimist_hits_5': 'both.pessimistic.hits_at_5',
               'b_realist_hits_5': 'both.realistic.hits_at_5',
               'b_optimist_hits_5': 'both.optimistic.hits_at_5',
               'h_pessimist_hits_10': 'head.pessimistic.hits_at_10',
               'h_realist_hits_10': 'head.realistic.hits_at_10',
               'h_optimist_hits_10': 'head.optimistic.hits_at_10',
               't_pessimist_hits_10': 'tail.pessimistic.hits_at_10',
               't_realist_hits_10': 'tail.realistic.hits_at_10',
               't_optimist_hits_10': 'tail.optimistic.hits_at_10',
               'b_pessimist_hits_10': 'both.pessimistic.hits_at_10',
               'b_realist_hits_10': 'both.realistic.hits_at_10',
               'b_optimist_hits_10': 'both.optimistic.hits_at_10'}

NODEPIECE_STRATEGY = {"degree": 0.4, "pagerank": 0.4, "random": 0.2}
TEST_STRATEGY = {"degree": 1.0, "pagerank": 0.0, "random": 0.0}
DEFAULT_NUM_ANCHORS = 50
DEFAULT_NUM_PATHS = 20

