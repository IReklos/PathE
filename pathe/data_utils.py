
import os
import glob
import logging
import itertools
import contextlib
from typing import List, Dict

import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset, random_split

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """
    Context manager seeding the NumPy PRNG with the specified seed and restoring
    the state afterward (when exiting from the context manager). All parameters
    provided are used to create a hash that will be used as the context seed.
    """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def local_seed(seed, *addl_seeds):
    """
    Context manager seeding both torch and numpy random generators and restoring
    their state after the contextualised (stochastic) operation has finished.
    This method generalises the ``numpy_seed()`` method. 
    """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state_np = np.random.get_state()
    state_torch = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state_np)
        torch.random.set_rng_state(state_torch)


def span_poisson(poisson_lambda):
    """
    Creates a Poisson distribution given a lambda. A categorical distribution
    is returned through torch, allowing to sample from it.
    """
    lambda_to_the_k = 1
    e_to_the_minus_lambda = math.exp(-poisson_lambda)
    k_factorial = 1
    ps = []
    for k in range(0, 128):
        ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
        lambda_to_the_k *= poisson_lambda
        k_factorial *= k + 1
        if ps[-1] < 0.0000001:
            break
    ps = torch.FloatTensor(ps)

    return torch.distributions.Categorical(ps)


def collate_tokens(
    values,
    padding_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """
    Convert a list of 1d tensors into a padded 2d tensor.

    Parameters
    ----------
    values : List[torch.tensor]
       A list of tensor that can have different lengths.
    padding_idx : int
        The index that will be used to encode padding tokens.
    eos_idx : int, optional
        The index that will be used to encode end of sequence tokens
    left_pad : bool, optional
        Whether padding is performed to the left, rather than to the right
    move_eos_to_beginning : bool, optional
        Whether sequences will be shifted to start with EOS, by default False
    pad_to_length : int, optional
        If greater than the longest sequence in the batch, this extends padding
    pad_to_multiple : int, optional
        If provided, padding is operated on multiples relatives to batch size
    pad_to_bsz : int, optional
        If provided, this will create a larger batch with fully padded sequences

    """
    size = max(v.size(0) for v in values)  # max sequence length in the batch
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    num_seq = len(values)  # tentative (vanilla) size for the sequence batch
    batch_size = num_seq if pad_to_bsz is None else max(num_seq, pad_to_bsz)
    batch = values[0].new(batch_size, size).fill_(padding_idx)  # mock batch

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, batch[i][size - len(v) :]
                    if left_pad else batch[i][: len(v)])
    
    return batch


def collate(
    samples : List[Dict],
    padding_idx : int,
    eos_idx : int = None,
    vocab : dict = None,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    """
    Collate samples from a dataset to create a batch for a Seq2Seq task.

    Parameters
    ----------
    samples : List[Dict]
        A list of dictionaries, one for each dataset record with the following
        entries: id (the identifier of the record), source, target
    padding_idx : int
        The index of the padding token in the vocabulary
    eos_idx : _type_
        The index of the end-of-sequence token in the vocabulary
    vocab : dict
        A dictionary containing a specification of the vocabulary
    left_pad_source : bool, optional
        _description_, by default False
    left_pad_target : bool, optional
        _description_, by default False
    input_feeding : bool, optional
        _description_, by default True
    pad_to_length : _type_, optional
        _description_, by default None

    Note: it is still not clear why ``eos_idx`` and ``vocab`` are  required,
          but not used in this function (possible ongoing development).
    Note: the `input_feeding` option is always expected for some reason (assert)
          suggesting that this code does not support other options here.
    """
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return collate_tokens(
            [s[key] for s in samples],
            padding_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] \
            if pad_to_length is not None else None,
    )
    # Sort by descending sequence (source) length
    id = torch.IntTensor([s["id"] for s in samples])
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)  # re-ordering the index
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens, target = None, None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s["target"]) for s in samples)

        if input_feeding:
            # We create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s["source"]) for s in samples)

    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        "nsequences": samples[0]["source"].size(0),
        "sort_order": sort_order,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens
    
    if "pos" in samples[0]:  # append pos information, if neeeded
        pos = merge(
            "pos",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["pos"]
            if pad_to_length is not None
            else None,
        )
        pos = pos.index_select(0, sort_order)
        batch["net_input"]["src_pos"] = pos

    if "relpred" in samples[0]:  # append rel prediction tensors, if needed
        batch["rel_prediction"] = {"cnt": [], "triples": []}
        for i, idx in enumerate(sort_order):
            record = samples[idx]["relpred"]
            batch["rel_prediction"]["cnt"].append(record.shape[0])
            batch["rel_prediction"]["triples"].append(torch.hstack(
                [torch.tensor([i]*record.shape[0]).unsqueeze(1), record]))
        # Stacking relation prediction tensors will merge all triples
        batch["rel_prediction"]["triples"] = torch.vstack(
            batch["rel_prediction"]["triples"])
        batch["rel_prediction"]["cnt"] = torch.LongTensor(
            batch["rel_prediction"]["cnt"])

    return batch


def collate_multipaths(
    samples : List[Dict],
    padding_idx : int,
    left_pad_seqs=False,
    pad_to_length=None,
):
    """
    Collate samples from a multi-path dataset to create a batch out of them.

    Parameters
    ----------
    samples : List[Dict]
        A list of dictionaries, one for each dataset record with the following
        entries: id (the identifier of the record), source, target.
    padding_idx : int
        The index of the padding token in the vocabulary.
    left_pad_seqs : bool, optional
        Whether to implement left padding on the sequences.
    pad_to_length : int, optional
        Extra padding is performed if needed, by default None.

    """
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        # First merge all sequences/tensors in a single list
        if isinstance(samples[0][key], list):
            seqs = list(itertools.chain.from_iterable([s[key] for s in samples]))
        else:  # sequence are not nested so do simple merge
            seqs = [s[key] for s in samples]
        seq_lenghts = torch.LongTensor([s.numel() for s in seqs])

        return collate_tokens(
            seqs,
            padding_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        ), seq_lenghts

    ent_paths, ent_paths_len = merge(
        "ent_paths",
        left_pad=left_pad_seqs,
        pad_to_length=pad_to_length["ent_paths"] \
            if pad_to_length is not None else None,
    )
    rel_paths, _ = merge(
        "rel_paths",
        left_pad=left_pad_seqs,
        pad_to_length=pad_to_length["rel_paths"] \
            if pad_to_length is not None else None,
    )
    # id = torch.IntTensor(list(itertools.chain.from_iterable(
    #     [[s["id"]] * len(s["ent_paths"]) for s in samples])))
    batch = {
        "id": torch.IntTensor([s["id"] for s in samples]),
        "ppt": torch.IntTensor([len(s["ent_paths"]) for s in samples]),
        "net_input": {
            "ent_paths": ent_paths,
            "rel_paths": rel_paths,
            "ent_lengths": ent_paths_len,
            "head_idxs": torch.IntTensor(list(itertools.chain.from_iterable(
                [s["head_indexes"] for s in samples]))),
        },
    }

    if "relation" in samples[0]:
        batch["target"] = torch.LongTensor([s["relation"] for s in samples])

    if "tail_indexes" in samples[0]:
        batch["net_input"]["tail_idxs"] = torch.IntTensor(
            list(itertools.chain.from_iterable(
                [s["tail_indexes"] for s in samples])))

    if "path_origins" in samples[0]:
        batch["net_input"]["path_origins"] = torch.IntTensor(
            list(itertools.chain.from_iterable(
                [s["path_origins"] for s in samples])))

    if "ori_triple" in samples[0]:
        batch["ori_triple"] = torch.stack([s["ori_triple"] for s in samples])
    
    if "pos" in samples[0]:
        batch["net_input"]["pos"], _ = merge(
            "pos",
            left_pad=left_pad_seqs,
            pad_to_length=pad_to_length["pos"] \
                if pad_to_length is not None else None,
        )

    return batch

################################################################################
# Some dataset utilities for splitting/partitioning and general factories
# ------------------------------------------------------------------------------

def split_datasets(dataset, split_pcts=[.8*.8, .8*.2, .2]):
    """
    Returns the training, validation and test sets/splits from a random
    partitioning of the given dataset using the split percentages.

    Parameters
    ----------
    dataset : torch.dataset)
        The main dataset to be split;
    split_pct : List[int]
        How to split the dataset for training, validation and test sets.
    
    """
    assert(1 < len(split_pcts) <= 3 and 1-1e-5 < sum(split_pcts) < 1+1e-5), \
        "The split percentages are too few or too many, or do not sum to 1."
    # Compute the actual size of each dataset split: the num of data points
    tr_samples = int(len(dataset) * split_pcts[0])
    va_samples = int(len(dataset) * split_pcts[1])
    te_samples = len(dataset) - (tr_samples + va_samples)
    # Partitioning the original dataset
    train_set, valid_set, test_set = random_split(
        dataset, [tr_samples, va_samples, te_samples])

    return train_set, valid_set, test_set


def partition_dataset(dataset, *partition_idxs):
    """
    Produces subsets of the dataset from the given partitions, where each
    each of them is provided as a list of partiion indices.

    Parameters
    ----------
    dataset : torch.Dataset
        A dataset that needs to be split based on the given indices.
    partition_idxs : List[List]
        A arbitrary number of dataset indices identifying the partitions.

    Notes
    -----
    - Still need to add the original FAIR/BART perturbation strategy
    - Still need to parameterise the composition pcts/probs

    """
    return (Subset(dataset, idxs) for idxs in partition_idxs)


def listify_columns(path_df: pd.DataFrame, *cols):
    """
    Replaces space-separated values in the specified columns with lists.
    """
    if len(cols) == 0:  # use all columns
        cols = list(path_df.columns)
    for col in cols:  # for each space-sep path
        path_df[col] = path_df[col].apply(
                lambda x: [int(i) for i in x.split()])


def load_unrolled_setup(folder_path: str, path_setup="20_20"):
    # Create file name patterns for expected files
    erpath_pattern = "_paths.csv"

    paths_files = []
    # Read the CSV files based on the patterns
    for subset_file in os.listdir(folder_path):
        if subset_file.endswith(erpath_pattern) \
            and subset_file.startswith(path_setup):
            paths_files.append(os.path.join(folder_path, subset_file))

    if len(paths_files) == 0:
        raise FileNotFoundError(f"No paths file found in {folder_path}")
    elif len(paths_files) > 1:
        logger.warning(f"Multiple paths found: using {paths_files[0]}")
    paths_files = paths_files[0]  # safe to use first available

    relcontext_path = os.path.join(folder_path, "relational_context.csv")
    if not os.path.isfile(relcontext_path):
        raise FileNotFoundError(relcontext_path)

    relneighbour_path = os.path.join(folder_path, "rel_neighbors.csv")
    if not os.path.isfile(relneighbour_path):
        raise FileNotFoundError(relneighbour_path)

    return paths_files, relcontext_path, relneighbour_path


def load_triple_tensors(train_path, val_path, test_path):
    """
    A function that loads the triple tensors

    Parameters
    ----------
    train_path : 
        The path to the train directory
    val_path :
        The path to the validation directory
    test_path :
        The path to the test directory

    Returns
    -------

    """
    train_triples = torch.load(os.path.join(train_path, 'triples.pt'))
    val_triples = torch.load(os.path.join(val_path, 'triples.pt'))
    test_triples = torch.load(os.path.join(test_path, 'triples.pt'))
    return train_triples, val_triples, test_triples


def load_corrupted_triples_from_dir(path: str):
    """
    Loads a dump of positive/negative triples from 1 or more `.pt` chunks.

    Parameters
    ----------
    path : str
        File system path to the folder containing a dump of pos/neg triples.

    Returns
    -------
    triple_dump : torch.tensor
        A tensor of positive triples followed by negatives triples from a dump.

    """
    if not os.path.exists(path):
        raise ValueError("The path provided is not valid.")
    # Get all files from dump folder and split suffix
    files = os.listdir(path)
    corrupted_pt_files = []
    for file in files:
        suffix = file.split(".")[-1]
        filename = file.split(".")[0]
        if suffix == "pt" and "corrupted" in filename:
            corrupted_pt_files.append(file)
    assert len(corrupted_pt_files) > 0
    # Prepare to merge all tensors after reading them
    ordered_tensors = [0] * len(corrupted_pt_files)
    for f in corrupted_pt_files:
        logger.info(f"Loading triple dump: {f}")
        tensor = torch.load(os.path.join(path, f))
        index = int(f.split("_")[0])
        if (index < len(ordered_tensors)) \
            and type(ordered_tensors[index]) is int:
            ordered_tensors[index] = tensor
        else:
            raise Exception("Multiple files of the same part of the "
                            "dataset or a part is missing.")
    corruptions = torch.cat(ordered_tensors, dim=0)
    del(ordered_tensors)
    positive_triples = corruptions[:, 0, :]
    negative_triples = corruptions[:, 1:, :]
    negative_triples = negative_triples.reshape(-1, 3)
    return torch.concat([positive_triples, negative_triples])


def memmap_corrupted_triples_from_dir(path: str):
    """
    Loads a dump of positive/negative triples from a single numpy chunk.

    Parameters
    ----------
    path : str
        File system path to the folder containing a dump of pos/neg triples.

    Returns
    -------
    triple_dump : np.memmap
        A numpy memory map indexing the data stored on disk.

    """
    chunks = glob.glob(os.path.join(path, "*.mmp"))
    if len(chunks) > 1:
        raise ValueError("Single numpy memory map expected, more found")
    elif len(chunks) == 0:
        raise ValueError(f"No numpy memory map found in {path}")
    chunk = chunks[0]  # safe to assume one at this stage

    fname = os.path.splitext(os.path.basename(chunk))[0]
    dims = [int(i) for i in fname.split('_')[1:]]
    assert len(dims) == 1, "Expected one dim only"  # no. of triples

    return np.memmap(chunk, dtype='int32', mode='r', shape=(dims[0],3))
