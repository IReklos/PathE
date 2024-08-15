"""
Denoising datasets for the reconstruction task of PathER, where logic is moved to
collate function in order to streamline data loading operations from arbitrary
files. Collation is used to create batch by adding the perturbations.

"""
import math
import logging
import itertools
from collections import Counter
from typing import Dict, List, Iterator

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import RandomApply, RandomChoice, RandomOrder

import data_utils as du
import path_lib as plib
from utils import sample_or_repeat
from corruption import generate_negative_triples, CorruptLinkGenerator

logger = logging.getLogger(__name__)


class InconsistentVocabulary(Exception):
    pass  # raised when the input vocab is not consistent with the data


def make_rpred_tensor(entities_idx, entities_pos, rpred_dict,
                      no_relt=torch.tensor([-1])):
    """
    Create a tensor for a relation prediction task by creating all combinations
    of the entities provided, and indexing them based on their position.

    Note: mechanism for balancing out the number of no-rel predictions as a way
          to counter the class imbalance of the classification task.

    Parameters
    ----------
    entities_idx : torch.tensor
        The sequence of entities to combine for relation prediction.
    entities_pos : torch.tensor
        The position of the CLS tokens for each entity in ``entities_idx``.
    rpred_dict : dict
        A nested dictionary mapping heads, to tails, to their relations.
    no_rel : torch.tensor
        The torch tensor corresponding to the "no-relation" token.

    Returns
    -------
    rprediction_t : torch.tensor
        The relation prediction tensor, containing all possible relations among
        the given entities. Columns are organised as: [0] head_cls_idx,
        [1] tail_cls_idx, [2] relation_id (dataset encoding from a PathDataset),
        [3] head_id (original), [4] tail_id (original).
    """
    assert len(entities_idx) == len(entities_pos)

    entities_list = list(entities_idx.numpy())
    enpos = dict(zip(entities_list, entities_pos))

    choice = lambda x: x[torch.randperm(x.numel())[0]]
    rprediction_t = [[enpos[h], enpos[t], choice(rpred_dict[h].get(t, no_relt)),
                      h, t]  # to account for the original node ids
                     for h in entities_list for t in entities_list
                     if h in rpred_dict and h != t]

    rprediction_t = torch.tensor(rprediction_t, dtype=entities_idx.dtype)
    if no_relt == -1:  # remove negative triples if norelt token not given
        # Keep all rows whose relation col (-1, the last one) does not have -1,
        # which denotes the default no-relation token (no_relt)
        rprediction_t = rprediction_t[rprediction_t[:, 2] != -1]

    return rprediction_t


def sample_relational_context(in_relcontext, out_relcontext, num_samples,
                              merge=False):
    """
    Randomly sample from the incoming and the outgoing relational contexts,
    either separately (two tensors are returned) or from both of them.

    Parameters
    ----------
    in_relcontext : torch.tensor
        The incoming relational context, given as a tensor of relation ids.
    out_relcontext : torch.tensor
        The outgoing relatioonal context, given as a tensor of relation ids.
    num_samples : int
        The number of unique tokens to sample from the contexts.
    merge : bool
        Whether sampling is done on the merged tensor or not.
    
    Note: we may want to parameterise the sampling function below, to minimise
          the impact of 1-to-many relations.
    """

    def sample(sampled_tensor: torch.tensor, num_samples: int, unique=True):
        # sampled_tensor = torch.tensor(sampled_tensor)  # make sure its a tensor
        sampled_tensor = sampled_tensor[torch.randperm(len(sampled_tensor))]
        sampled_tensor = torch.unique(
            sampled_tensor) if unique else sampled_tensor
        # Now we are safe to return min(``num_samples``, len(``sampled_tensor``)
        return sampled_tensor[:num_samples]

    if merge:  # return a single sample from both the inc and outg contexts
        return sample(torch.cat((in_relcontext, out_relcontext)), num_samples)
    in_relcontext = sample(in_relcontext, num_samples)
    out_relcontext = sample(out_relcontext, num_samples)
    return in_relcontext, out_relcontext


def sample_relational_context_with_oversampling(in_relcontext, out_relcontext,
                                                num_samples,
                                                merge=False):
    """
    Randomly sample from the uniques of the incoming and the outgoing
    relational contexts, either separately (two tensors are returned)
     or from both of them and oversample relation based on their frequency

    Parameters
    ----------
    in_relcontext : torch.tensor
        The incoming relational context, given as a tensor of relation ids.
    out_relcontext : torch.tensor
        The outgoing relatioonal context, given as a tensor of relation ids.
    num_samples : int
        The number of unique tokens to sample from the contexts.
    merge : bool
        Whether sampling is done on the merged tensor or not.

    Note: we may want to parameterise the sampling function below, to minimise
          the impact of 1-to-many relations.
    """

    def sample(sampled_tensor: torch.tensor, num_samples: int, unique=True):
        # get unique relations and their counts
        sampled_tensor, counts = torch.unique(sampled_tensor,
                                              return_counts=True)
        # get sorted indices of the counts (same for relations), to sample
        # preferentially the higher frequency relations
        _, sorted_counts = torch.sort(counts, descending=True)
        # sort the tensors and sample
        sampled_tensor = sampled_tensor[sorted_counts][:num_samples]
        counts = (counts.float() / torch.sum(counts))
        counts = counts[sorted_counts][:num_samples]

        # calculate the number of times each relations should be repeated
        # oversampling = ((counts.float() / torch.sum(counts)) * (
        #         2*num_samples)).round().to(torch.int32)
        oversampling = torch.ceil(counts * (2 * num_samples)).to(torch.int32)
        # repeat each relation
        sampled_tensor = torch.repeat_interleave(sampled_tensor, oversampling)
        # Now we are safe to return min(``num_samples``, len(``sampled_tensor``)
        return sampled_tensor

    if merge:  # return a single sample from both the inc and outg contexts
        return sample(torch.cat((in_relcontext, out_relcontext)), num_samples)
    in_relcontext = sample(in_relcontext, num_samples)
    out_relcontext = sample(out_relcontext, num_samples)
    return in_relcontext, out_relcontext


def create_vocabulary(paths: List[List[int]], xtokens: List[str]):
    """
    Create a vocabulary based on the relations retrieved from the path store.
    This will provide an encoding for both the relations and the special tokens,
    which will be prepended in the mapping (they offset the rel. mapping). Note
    that the PADding token is always included to extend the ``xtokens``.

    Parameters
    ----------
    paths : List[List[int]]
        A list containing relational paths, encoded as a list of integers.
    xtokens : List[str]
        The special tokens to add to the vocabulary, which will offset idxs.

    Note: at the moment, it also provides statistics to potentially filter out
        less frequent relations, but this is not yet reflected in the encoding.
    """
    # Counting relation occurrences across all paths and relative freqs
    relation_cnt = Counter()
    for path in tqdm(paths):
        relation_cnt.update(path)
    relation_tot = relation_cnt.total()
    relation_pct = {k: v / relation_tot for k, v in relation_cnt.items()}
    # self.start_nodes_vocab = sorted(np.unique(sample_df['start_node']))
    # self.end_nodes_vocab = sorted(np.unique(sample_df['end_node']))
    relation_vocab = sorted(relation_cnt)
    min_relation_idx, max_relation_idx = \
        min(relation_vocab), max(relation_vocab)
    expected_relation_vocab = list(
        range(min_relation_idx, max_relation_idx + 1))
    if relation_vocab != expected_relation_vocab:
        missing = set(expected_relation_vocab) - set(relation_vocab)
        logger.warn("Relation set may not be complete, sample more paths before"
                    f" training! Removing unseen relations for now: {missing}")
    print(f"Relation size: {len(relation_vocab)} min {min_relation_idx}")
    # Saving token encoder and decoder: from tokens to indices and v.v.
    tokens_to_idxs = {t: idx for idx, t in enumerate(
        ["PAD"] + xtokens + relation_vocab)}

    return relation_vocab, tokens_to_idxs


def check_vocabulary(paths: List[List[int]], vocabulary: List):
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


class PathDataset(Dataset):

    def __init__(self, path_store: str, seed: int = 46, transform=None,
                 xtokens: List[str] = ["MSK", "CLS", "SEP", "UKN"],
                 tokens_to_idxs: Dict[int, int] = None):
        """
        Generalisation to describe datasets of network paths given in a CSV file
        and encoded as sequence of relations, given by their index. Although the
        encoding is re-initialised to start from 0, relation indexes are assumed
        to be contiguous: spanning all integers from a certain range. If this
        does not happen, the code assumes that more paths should be sampled.

        Extra indices can be used on request to denote special tokens such as:
        - [PAD] used for creating batches with same-length sequences
        - [MSK] used for masked sequence modelling (input sequences only)
        - [CLS] used for additional learning strategies at the subsequence level
        - [SEP] used to denote sub-sequence separators (e.g. full stop in text)
        - [UKN] for encoding rare, uncommon, or unformatted observations

        Parameters
        ----------
        path_store : str
            Path to the CSV file storing Knowledge Graph paths, where the first
            two columns denote the indexes of the starting and the ending node
            of each path, whereas the third column holds the sequence of crossed
            relation indexes in the path, separated by a space.
        transform : fn, optional
            A transformation to apply to paths before they are returned.
        xtokens : List[str]
            A list of extra/special tokens whose index will preceed the mapping.

        """
        sample_df = pd.read_csv(path_store)
        print(f"Found {len(sample_df)} paths from {path_store}")
        # Split the path string on spaces and convert to list of int (edge ids)
        # and compute path len (which can help sorting sequences prior to batch)
        sample_df['relation_path'] = sample_df['relation_path'].apply(
            lambda x: [int(i) for i in x.split()])
        sample_df['path_len'] = sample_df['relation_path'].apply(
            lambda x: len(x))

        if tokens_to_idxs is not None:
            logger.info(f"Using given encoding: {len(tokens_to_idxs)} tokens")
            self.tokens_to_idxs = tokens_to_idxs  # TODO Sanity check
            assert all([xt in self.tokens_to_idxs for xt in xtokens])
            self.no_relations = len(tokens_to_idxs) - len(xtokens) - 1
        else:  # otherwise, recompute all the encoding and vocabulary
            logger.info(f"Creating new vocabulary from {path_store}")
            relation_vocab, self.tokens_to_idxs = create_vocabulary(
                list(sample_df['relation_path']), xtokens=xtokens)
        # Decoding map: from idxs back to the original sequence tokens
        self.idxs_to_tokens = {idx: t for t, idx in self.tokens_to_idxs.items()}

        self.epoch = None
        self.transform = transform
        self.paths = sample_df.iloc[:, 2].values
        self.vocab_size = len(self.tokens_to_idxs)
        self.seed = seed if seed is not None else torch.seed()

    def __len__(self):
        return len(self.paths)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, idx):
        tokens = self.paths[idx]  # path sequence - List of int edge ids
        tokens = torch.LongTensor([self.tokens_to_idxs[t] for t in tokens])
        source, target = tokens, tokens.clone()

        if self.transform is not None:
            with du.numpy_seed(self.seed, self.epoch, idx):
                source = self.transform(source)  # w.r.t. perturb strategy

        return {"id": idx, "source": source, "target": target}


class UnrolledPathDataset(Dataset):

    def __init__(self, path_store: str,
                 relneighb_store: str,
                 relcontext_store: str,
                 relcontext_size: int,
                 tokens_to_idxs: Dict[int, int] = None,
                 transform_relcontext=None,
                 transform_relations=None,
                 transform_entities=None,
                 seed: int = 46, ):
        """
        A PathDatest where paths hold the relational context of nodes traversed.

        This dataset use a predefined set of special tokens:
        - [PAD] used for creating batches with same-length sequences
        - [MSK] used for masked sequence modelling (input sequences only)
        - [CLS] used for entities embeddings w.r.t the relational context
        - [SEP] to separate and denote bridging relations from context relations

        Parameters
        ----------
        path_store : str
            Path to the CSV file storing paths, where the first column contains
            the relations in the path, and the second holds the traversed nodes.
        relneighb_store : str
            Path to the CSV file holding the relations that occur between every
            pair of entity in the knowledhe graph (a summary of local info).
        relcontext_store : str
            Path to the CSV file storing the relational context for each entity,
            where `node_id` denotes the entity, `direction` holds the type of
            context (either `in` or `out` for incoming and outgoing relations),
            and `edges` contains the whole list of relations (with repetitions).
        transform : fn, optional
            A transformation to apply to paths before they are returned.
        xtokens : List[str]
            A list of extra/special tokens whose index will preceed the mapping.

        """
        xtokens = ["MSK", "CLS", "SEP"]  # NR
        pathstore_df = pd.read_csv(path_store)
        relcontext_df = pd.read_csv(relcontext_store)
        relneighb_df = pd.read_csv(relneighb_store)
        # Step 1: Read paths and convert them to plain lists
        print(f"Found {len(pathstore_df)} paths from {path_store}")
        du.listify_columns(pathstore_df)  # space-separated rels --> lists
        du.listify_columns(relcontext_df, "edges")  # same for rel context
        du.listify_columns(relneighb_df, "Relations")  # and same for neighbours

        # Step 2: Load or build the vocabulary using the relational context
        if tokens_to_idxs is not None:
            logger.info(f"Using given encoding: {len(tokens_to_idxs)} tokens")
            if not check_vocabulary(relcontext_df["edges"],
                                    tokens_to_idxs.keys()):
                raise InconsistentVocabulary("Missing tokens in vocabulary")
            self.tokens_to_idxs = tokens_to_idxs
        else:  # otherwise, recompute all the encoding and vocabulary
            logger.info(f"Creating new vocabulary from {path_store}")
            relation_vocab, self.tokens_to_idxs = create_vocabulary(
                list(relcontext_df["edges"]), xtokens=xtokens)
        # Decoding map: from idxs back to the original sequence tokens
        self.idxs_to_tokens = {idx: t for t, idx in self.tokens_to_idxs.items()}

        # Step 3: Encoding and building tensors for relations and entities
        encode = lambda x: torch.LongTensor([self.tokens_to_idxs[r] for r in x])
        self.relation_paths = [encode(p) for p in pathstore_df["relation_path"]]
        self.entity_paths = [torch.LongTensor(p)  # not encoded!
                             for p in pathstore_df["entity_path"]]

        # Step 4:  Convert relational context to a tensor dict
        relcontexts = {}  # {node_id: {direction: torch.tensor(relations)}}
        for i, record in relcontext_df.iterrows():
            if record["node_id"] not in relcontexts:
                relcontexts[record["node_id"]] = {}
            relcontexts[record["node_id"]][record["direction"]] = \
                encode(record["edges"])  # needed for sampling

        # Step 5: Convert relational neighbour to a tensor dict
        relneighbours = {}  # {node_id: {node_id: torch.tensor(relations)}}
        for i, record in relneighb_df.iterrows():
            if record["Head"] not in relneighbours:
                relneighbours[record["Head"]] = {}
            relneighbours[record["Head"]][record["Tail"]] = \
                encode(record["Relations"])  # needed for sampling

        self.epoch = None
        self.relcontexts = relcontexts
        self.relneighbours = relneighbours
        self.relcontext_size = relcontext_size
        self.vocab_size = len(self.tokens_to_idxs)
        self.no_relations = self.vocab_size - len(xtokens) - 1  # PAD
        self.transform_rc = transform_relcontext
        self.transform_re = transform_relations
        self.transform_en = transform_entities
        self.seed = seed if seed is not None else torch.seed()

    def __len__(self):
        return len(self.relation_paths)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def create_unrolled_path(self, relations, entities, relcontext_size,
                             local_seed=1, perturb=True):
        """
        Create and assemble an unrolled path given relations and entities.
        """
        # FIXME Understand if we can just encode these for all records
        clst = torch.tensor([self.tokens_to_idxs["CLS"]], dtype=entities.dtype)
        sept = torch.tensor(self.tokens_to_idxs["SEP"], dtype=clst.dtype)
        mskt = torch.tensor(self.tokens_to_idxs["MSK"], dtype=clst.dtype)
        empt = torch.tensor([], dtype=clst.dtype)  # for nodes with no rel
        mcon = torch.tensor([mskt, clst, mskt],
                            dtype=clst.dtype)  # mskd context

        relational_contexts = {}  # node_id → relational context
        for node in entities:  # sampling relational contexts for nodes
            node_relcontext = self.relcontexts[node.item()]  # full RC
            # logger.debug(f"Node {node} full RC: {node_relcontext}")
            in_relcontext, out_relcontext = sample_relational_context(
                node_relcontext.get("in", empt),
                node_relcontext.get("out", empt),
                relcontext_size,
            )
            # TODO Check that we have at least a context available
            if perturb and self.transform_rc is not None:  # local perturbations
                with du.numpy_seed(self.seed, self.epoch, local_seed):
                    in_relcontext = self.transform_rc(in_relcontext)
                    out_relcontext = self.transform_rc(out_relcontext)
            # XXX This would fail if the path contains loops (entity rep)
            relational_contexts[node.item()] = torch.concat(
                (in_relcontext, clst, out_relcontext))

        source_relations = relations.clone()
        if perturb and self.transform_re is not None:  # relation perturbations
            with du.numpy_seed(self.seed, self.epoch, local_seed):
                source_relations = self.transform_re(source_relations)

        source_entities = entities.clone()
        if perturb and self.transform_en is not None:  # entity perturbations
            with du.numpy_seed(self.seed, self.epoch, local_seed):
                source_entities = self.transform_en(source_entities)

        # So far, no support for deletion/insertion
        assert source_entities.numel() == entities.numel()
        assert source_relations.numel() == relations.numel()
        unrolled_path = []  # assembling the unrolled sequence
        unrolled_pos = []  # holding the true positions of tokens
        # Assembling the unrolled path via concatenation of RCs and bridges
        for i, (node, rel) in enumerate(zip(source_entities, source_relations)):
            rc_con = relational_contexts[node.item()] if node != mskt else mcon
            unrolled_path.append(rc_con)  # relational or masked context
            unrolled_path.append(torch.LongTensor([sept, rel, sept]))
            unrolled_pos += [i * 2] * len(rc_con) + [i * 2 + 1] * 3
        # Appending information of the last relational context in the sequence
        unrolled_path.append(relational_contexts[source_entities[-1].item()])
        unrolled_path = torch.concat(unrolled_path)
        unrolled_pos += [unrolled_pos[-1] + 1] * \
                        (len(unrolled_path) - len(unrolled_pos))
        # Creating torch tensors for the positionals and start from 1
        unrolled_pos = torch.tensor(unrolled_pos, dtype=unrolled_path.dtype) + 1

        return unrolled_path, unrolled_pos, source_entities

    def __getitem__(self, idx):
        # Retrieve the relational and entity paths from record index
        relations, entities = self.relation_paths[idx], self.entity_paths[idx]
        unrolled_path, unrolled_pos, source_entities = \
            self.create_unrolled_path(relations, entities,
                                      self.relcontext_size, local_seed=idx)
        # Final round: preparing the relation prediction tensor for CLS task
        clst = torch.tensor([self.tokens_to_idxs["CLS"]], dtype=entities.dtype)
        # FIXME This should also remove the MSKed relational context otw fails
        cls_pos = (unrolled_path == clst).argwhere().flatten()
        relpred = make_rpred_tensor(source_entities, cls_pos,
                                    self.relneighbours,
                                    torch.tensor([-2]))  # FIXME!!!

        return {"id": idx, "pos": unrolled_pos, "relpred": relpred,
                "source": unrolled_path, "target": relations}

    # Decoder expects the correct, original sequence of relations
    # Encoder wants the relation between each node for CLS objective
    # Encoder wants the full sequence with separators, etc.
    # Permutations need to remember which relation connects two nodes


class TriplePathDataset(Dataset):
    """
    A dataset for representing triples as paths: where heads and tails are
    replaced by an incoming and an outgoing path, respectively. The current
    implementation reuses a ``PathDataset`` to reuse path construction logic.

    Parameters
    ----------
    triplestore : torch.tensor
        A matrix holding (h, r, t) triples, following the original raw encoding.
    pathdataset : UnrolledPathDataset
        The unrolled path dataset from which paths will be drawn for concat.
    relcontext_size : int
        The size of the relational contexts to sample for entities.
    trim_context_len : int
        If provided, context (entity-relation) paths are trimmed to the given
        length; for example, if set to 5, it trims 3 entities and 2 relations.
    """

    def __init__(self, triplestore, pathdataset, relcontext_size,
                 trim_context_len=101, num_negatives=0, triple_corruptor=None):

        if trim_context_len % 2 == 0:  # expecting N entities + (N-1) relations
            raise ValueError("Context path length can only be odd")

        path_index = {"in": {}, "out": {}}
        for i in tqdm(range(len(pathdataset)), desc="H-T path indexing"):
            # Retrieving start and end node from entity paths
            start, end = pathdataset.entity_paths[i][[0, -1]]
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

        self.num_positive, self.num_negatives = len(triplestore), num_negatives
        if num_negatives > 0:  # This triggers the creation of negative samples
            logger.info(f"Creating {num_negatives} neg per {self.num_positive}")
            corruptions = triple_corruptor.get_filtered_corrupted_triples(
                triplestore, num_negatives)  # generate full corruption tensor
            positive_triples = corruptions[:, 0, :]
            negative_triples = corruptions[:, 1:, :]
            negative_triples = negative_triples.reshape(-1, 3)
            assert len(negative_triples) == self.num_positive * num_negatives \
                   or len(
                negative_triples) == self.num_positive * num_negatives * 2
            assert torch.all(positive_triples.unique(dim=0) == triplestore)
            triplestore = torch.concat([positive_triples, negative_triples])

        self.triplstore = triplestore
        self.path_index = path_index
        self.pathdataset = pathdataset
        self.relcontext_size = relcontext_size
        self.rel_cut = trim_context_len // 2
        self.ent_cut = self.rel_cut + 1
        self.dtype = pathdataset.entity_paths[0].dtype

    def __len__(self):
        return len(self.triplstore)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, idx):

        head, relation, tail = self.triplstore[idx]
        # Encoding the relation using the pathdataset mapping
        mskt = torch.tensor(self.pathdataset.tokens_to_idxs["MSK"],
                            dtype=self.dtype)
        relation_enc = self.pathdataset.tokens_to_idxs[relation.item()]
        relation_enc = torch.tensor(relation_enc, dtype=mskt.dtype)
        # print(f"Triple: {head}, {relation}({relation_enc}), {tail}")

        # First step is to assemble the template of the path
        choice = lambda x: np.random.choice(x, 1)[0]
        entities, relations = [], []
        # Retrieving an incoming path, if present, or use head
        if head.item() in self.path_index["in"]:
            # print(self.path_index["in"][head.item()])
            in_path_idx = choice(self.path_index["in"][head.item()])
            entities += self.pathdataset.entity_paths[in_path_idx][
                        -self.ent_cut:]
            relations += self.pathdataset.relation_paths[in_path_idx][
                         -self.rel_cut:]
        else:  # just include the node
            entities.append(head)

        relations.append(relation_enc)
        relation_loc, head_loc = len(relations) - 1, len(entities) - 1
        # Retrieving an outgoing path, if present, or use tail
        if tail.item() in self.path_index["out"]:
            out_path_idx = choice(self.path_index["out"][tail.item()])
            entities += self.pathdataset.entity_paths[out_path_idx][
                        :self.ent_cut]
            relations += self.pathdataset.relation_paths[out_path_idx][
                         :self.rel_cut]
        else:  # just include the node
            entities.append(tail)

        entities = torch.tensor(entities, dtype=mskt.dtype)
        relations = torch.tensor(relations, dtype=mskt.dtype)
        relations_tmask = relations.clone()
        relations_tmask[relation_loc] = mskt

        unrolled_path, unrolled_pos, _ = self.pathdataset.create_unrolled_path(
            relations_tmask, entities, self.relcontext_size, perturb=False)

        clst = torch.tensor(self.pathdataset.tokens_to_idxs["CLS"],
                            dtype=mskt.dtype)
        cls_pos = (unrolled_path == clst).argwhere().flatten()

        relpred = torch.tensor([[cls_pos[head_loc], cls_pos[head_loc + 1],
                                 relation_enc, head, tail]], dtype=mskt.dtype)

        return {"id": idx, "pos": unrolled_pos, "relpred": relpred,
                "source": unrolled_path, "target": relations, "ents": entities}


class NegativeTripleSampler(Sampler[int]):
    """
    Samples positive and negative triples from a TriplePathDataset.

    Parameters
    ----------
    num_positives : int
        Total number of positive triples in the dataset.
    k : int
        Number of negative triples (corruptions) per positive.
    """

    def __init__(self, num_positives, k) -> None:
        self.num_positives = num_positives
        self.k = k  # negatives per positive

    def __iter__(self) -> Iterator[int]:
        # Create interleaved triple indexes: [p_i, n_i1, ... n_ik]
        indexes = [[i] + list(range(self.num_positives + self.k * i,
                                    self.num_positives + self.k * i + self.k))
                   for i in range(self.num_positives)]

        yield from itertools.chain.from_iterable(indexes)

    def __len__(self) -> int:
        return self.num_positives * (self.k + 1)


################################################################################
# Multi-path datasets
# ------------------------------------------------------------------------------

class SimplePathDataset(Dataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                 parallel = False,
                 neg_triple_store = None,
                 ):
        """
        A base class for path-based datasets, wrapping the basic functionalities
        including pre-processing, vocabulary generation, path indexing, encoding
        and negative generation.

        Parameters
        ----------
        path_store : str
            Path to the CSV file storing paths, where the first column contains
            the relations in the path, and the second holds the traversed nodes.
        relcontext_store : str
            Path to the CSV file storing the relational context for each entity,
            where `node_id` denotes the entity, `direction` holds the type of
            context (either `in` or `out` for incoming and outgoing relations),
            and `edges` contains the whole list of relations (with repetitions).
        triplestore : torch.tensor
            A matrix holding (h, r, t) triples, following the original encoding.
        context_triple_store : torch.tensor
            If provided, these triples will be used for constructing paths,
            rather than using (as default option) those in ``triplestore``. 
        tokens_to_idxs : Dict[int, int]
            Mapping dictionary for encoding tokens to proper idxs.
        maximum_triple_paths : int
            The maximum number of paths per triple that will be returned.
        seed : int
            A seed to control the sthocasticity of the dataset.

        """
        xtokens = ["MSK"]  # the special tokens that will be reserved
        relcontext_df = pd.read_csv(relcontext_store)
        du.listify_columns(relcontext_df, "edges") # space-sep rels --> lists

        if isinstance(path_store, str):
            # Step 1: Read paths and convert them to plain lists
            pathstore_df = pd.read_csv(path_store)
            print(f"Found {len(pathstore_df)} paths from {path_store}")
            du.listify_columns(pathstore_df)  # space-separated vals --> lists
            # Step 2: Encoding and building tensors for relations and entities
            to_tensor = lambda x: [torch.IntTensor(t) for t in x]
            self.relation_paths = to_tensor(pathstore_df["relation_path"])
            self.entity_paths =  to_tensor(pathstore_df["entity_path"])
            self.path_index = plib.create_path_indexing(self.entity_paths)
        else:  # this assumes that the pathstore is a tuple
            self.relation_paths, self.entity_paths, self.path_index = path_store

        # Step 2: Load or build the vocabulary using the relational context
        if tokens_to_idxs is not None:
            logger.info(f"Using given encoding: {len(tokens_to_idxs)} tokens")
            if not check_vocabulary(relcontext_df["edges"], tokens_to_idxs.keys()):
                raise InconsistentVocabulary("Missing tokens in vocabulary")
            self.tokens_to_idxs = tokens_to_idxs
        else:  # otherwise, recompute all the encoding and vocabulary
            logger.info(f"Creating new vocabulary from {path_store}")
            relation_vocab, self.tokens_to_idxs = create_vocabulary(
                list(relcontext_df["edges"]), xtokens=xtokens)
        # Decoding map: from idxs back to the original sequence tokens
        self.idxs_to_tokens = {idx: t for t, idx in self.tokens_to_idxs.items()}

        self.num_pos, self.num_neg = len(triple_store), num_negatives
        if num_negatives > 0:  # triggers the extension of the triple store
            if neg_triple_store is None:  # create new negatives
                triple_store = generate_negative_triples(
                    triple_store, num_negatives,
                    triple_corruptor=triple_corruptor, parallel=parallel)
            else:  # attempting to reuse precomputed negatives
                triple_store = neg_triple_store  # FIXME
                expected_dim = (self.num_pos * 2) * (num_negatives + 1)
                assert neg_triple_store.shape[0] == expected_dim, \
                    "Corrupted triplestore dim does not match the triplestore:"\
                    f" got {neg_triple_store.shape[0]}, expected {expected_dim}"

        self.epoch = None
        self.xtokens = xtokens
        self.triplestore = triple_store
        self.max_paths = maximum_triple_paths
        self.vocab_size = len(self.tokens_to_idxs)
        self.context_triple_store = context_triple_store \
              if context_triple_store is not None else triple_store
        self.no_relations = self.vocab_size - len(xtokens) - 1  # 1 for PAD
        self.seed = seed if seed is not None else torch.seed()

    def __len__(self):
        return len(self.triplestore)

    def set_epoch(self, epoch, **unused):
        # This is kept for compatibility with other path datasets
        self.epoch = epoch

    def encode_relations(self, relations):
        # Follows the encoding computed at construction time or reused
        if torch.is_tensor(relations):
            relations =  relations.tolist()
        return torch.IntTensor([self.tokens_to_idxs[r] for r in relations]) \

    def encode_entities(self, entities):
        # Recreates the encoding of relations for entities, although we are not
        # really encoding nodes; but rather prepare a consistent mapping
        if torch.is_tensor(entities):
            entities =  entities.tolist()
        return torch.IntTensor([r + len(self.xtokens) + 1 for r in entities])
    
    def fetch_path_context(self, index: int, context_type: str):
        """
        Returns the entity and relation path associated to a path template.

        Parameters
        ----------
        index : int
            Index of the path that will be fetched from the pathstore.
        context_type : str
            Type of context, either ``triple`` or ``path``.

        """
        if context_type == "path":
            entities = self.entity_paths[index]
            relations = self.relation_paths[index]
        elif context_type == "triple":
            triple = self.context_triple_store[index]
            entities = triple[[0, -1]]
            relations = triple[[1]]
        else:  # context type is not path nor triple, hence not supported
            raise ValueError(f"{context_type} not a supported context type")

        return entities, relations

    def __getitem__(self, index) -> dict:
        raise NotImplementedError("Abstract class needs extension")

    def dump_negatives(self, fname: str):
        """Dumps the negative triples to disk, depending on corruption type."""
        negative_start = self.num_pos * 2 \
            if self.triplestore[0] == self.triplestore[1] else self.num_pos
        negatives = self.triplestore[negative_start:]
        torch.save(negatives, fname)
    
    def _load_negatives(self, negatives: torch.tensor, corruptor):
        if isinstance(corruptor, CorruptLinkGenerator):
            self.triplestore = self.triplestore.repeat_interleave(2)
        self.triple_store = torch.vstack([self.triple_store, negatives])


class MultiPathDataset(SimplePathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                 parallel=False,
                 neg_triple_store = None,
                ):
        """
        Adds methods and attributes to deal with multiple paths.
        """
        super().__init__(path_store,relcontext_store, triple_store,
                         context_triple_store, tokens_to_idxs,
                         maximum_triple_paths, num_negatives,
                         triple_corruptor, seed, parallel, neg_triple_store)
        # Distributing path budget across entities (H, T) and contexts (in, out)
        self.ppe = maximum_triple_paths // 2  # paths per entity
        self.ppc = self.ppe // 2  # paths per entity context (in/out)

    def _create_inout_contextpaths(self, entity):
        """
        Creates contextualised paths by combining incoming and outgoing paths
        for a given entity, and updating positional informatioon.
        """
        et = lambda: torch.IntTensor([])
        in_paths, out_paths = plib.create_contextpaths(
            entity, self.path_index, self.context_triple_store)
        # Get maximum posible num paths and sample from each context
        max_ppe = min(self.ppe, max(len(in_paths), len(out_paths)))
        path_sample_a = sample_or_repeat(in_paths, max_ppe)
        path_sample_b = sample_or_repeat(out_paths, max_ppe)
        # assert len(path_sample_b) == len(path_sample_a)
        ent_paths, rel_paths, ht_idxs, er_pos = [], [], [], []
        for in_context, out_context in zip(path_sample_a, path_sample_b):

            if in_context is not None:
                in_ents, in_rels = self.fetch_path_context(*in_context)
                entity = in_ents[-1]
                in_ents = in_ents[:-1]
            else:  # no incoming context is available
                in_ents, in_rels = et(), et()

            if out_context is not None:
                out_ents, out_rels = self.fetch_path_context(*out_context)
                entity = out_ents[0]
                out_ents = out_ents[1:]
            else:  # no outgoing context is available
                out_ents, out_rels = et(), et()

            entix = len(in_ents)  # position of the entity in the path
            entities = torch.concat([in_ents, entity.unsqueeze(0), out_ents])
            relations = torch.concat([in_rels, out_rels])
            assert len(entities) == len(relations) + 1

            ht_idxs.append(entix)
            ent_paths.append(self.encode_entities(entities))
            rel_paths.append(self.encode_relations(relations))
            er_pos.append(plib.get_entfocused_positionals(
                entix*2, len(entities)*2-1) + 1)  # +1 to avoid PAD

        return ent_paths, rel_paths, ht_idxs, er_pos


class ProtopathDataset(SimplePathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                ):
        """
        One triple, several paths traversing the same triple for context. These
        are constructed by considering the various combinations of incoming and
        outgoing paths (or triples, if paths are not available) with respect to
        the head and tail, respectively. Rather than explicitly storing paths,
        only the actual indexes are kept in memory for efficiency.
        """
        super().__init__(path_store,relcontext_store, triple_store,
                         context_triple_store, tokens_to_idxs,
                         maximum_triple_paths, num_negatives,
                         triple_corruptor, seed)
        # Optionally creates protopaths if caching is enabled
        self.protopath_caching = False # FIXME to parameterise
        if self.protopath_caching:
            self._create_triple_protopaths()

    def _create_triple_protopaths(self):
        # Create protopaths only for the unique combinations of heads-tails
        self.protopath_map = {}  # indexed by head-tail
        for i in tqdm(range(self.triplestore.shape[0]), desc="Protopaths gen"):
            # Retrieving the triple and creating
            head, rel, tail = self.triplestore[i]
            protopath_hash = f"{head}_{tail}"
            if protopath_hash not in self.protopath_map:
                self.protopath_map[protopath_hash] = plib.create_protopaths(
                    head, tail, self.path_index, self.context_triple_store)

    def instantiate_protopath(self, triple, propath):
        """
        Generate an actual entity-relation path given a protopath, indexing the
        incoming and the outgoing context, if available; or use triple only.

        Parameters
        ----------
        triple : torch.tensor
            A triple to contextualise/extend to form a path.
        propath : tuple
            A tuple specifying the idx and type of incoming and outgoing context

        """
        to_tensor = lambda x: torch.tensor(x, dtype=triple.dtype)
        # Fetch the incoming (head) and the outgoing (tail) context, if avail
        entities_in, relations_in = self.fetch_path_context(
            propath[0], propath[1]) if propath[0] is not None \
                else (to_tensor([triple[0]]), to_tensor([]))
        entities_out, relations_out = self.fetch_path_context(
            propath[2], propath[3]) if propath[2] is not None \
                else (to_tensor([triple[-1]]), to_tensor([]))

        # print(entities_in, relations_in)
        # print(entities_out, relations_out)
        assert entities_in[-1] == triple[0] and entities_out[0] == triple[-1]
        head_idx = len(entities_in) - 1  # tail_idx = head_idx + 1
        # Merging incoming with outgoing contexts with the bridging triple
        ents = torch.concat([entities_in[:-1], triple[[0,-1]], entities_out[1:]])
        rels = torch.concat([relations_in, triple[[1]], relations_out])

        return ents, rels, head_idx

    def __getitem__(self, index) -> dict:
        """
        Get a multi-paths contextualisation of the i-th triple in the dataset.
        This implementation either reuses protopath caching, or creates otf. 
        """
        triple = self.triplestore[index]
        head, rel, tail = triple
        protopaths = self.protopath_map[f"{head}_{tail}"] \
            if self.protopath_caching else plib.create_protopaths(
                head, tail, self.path_index, self.context_triple_store)

        with du.local_seed(self.seed, self.epoch, index):
            # Sampling max N before translating protopaths
            protopaths = [protopaths[i] for i in
                          torch.randperm(len(protopaths))[:self.max_paths]]

        ent_paths, rel_paths, head_idxs = [], [], []
        for protopath in protopaths:
            ents, rels, head_idx = self.instantiate_protopath(triple, protopath)
            # Encode both relations and entities, and mask the bridge relation
            rels = self.encode_relations(rels)
            rels[head_idx] = self.tokens_to_idxs["MSK"]
            rel_paths.append(rels)
            ent_paths.append(self.encode_entities(ents))
            head_idxs.append(head_idx)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": head_idxs, "relation": self.tokens_to_idxs[rel.item()],
        }


class TripleEntityMultiPathDataset(MultiPathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 context_triple_store: torch.tensor = None,
                 tokens_to_idxs: Dict[int, int] = None,
                 maximum_triple_paths = 50,
                 num_negatives: int = 0,
                 triple_corruptor = None,
                 seed: int = 46,
                 parallel=False,
                 neg_triple_store = None,
                ):
        """
        A triple-centric dataset where the given head and tail are used to grab
        paths traversing such entities, individually. The triple item is then
        defined as a balanced set of paths starting or ending from/at heads and
        tails; without introducing any particular contraint. This comes at the
        benefit of avoiding combinatorial issues of the TripleMultiPathDataset.
        """
        super().__init__(path_store,relcontext_store, triple_store,
                         context_triple_store, tokens_to_idxs,
                         maximum_triple_paths, num_negatives,
                         triple_corruptor, seed, parallel, neg_triple_store)

    def _getitem_separate(self, index) -> dict:
        """
        Get a multi-paths contextualisation of the i-th triple in the dataset,
        where incoming and outgoing paths are kept independent.
        """
        triple = torch.as_tensor(self.triplestore[index])
        head, rel, tail = triple
        # Retrieving incoming and outgoing paths/triples per entity
        in_hpaths, out_hpaths = plib.create_contextpaths(
            head, self.path_index, self.context_triple_store)
        in_tpaths, out_tpaths = plib.create_contextpaths(
            tail, self.path_index, self.context_triple_store)
        # Record the origin of each path (0 for H, 1 for T) capped at ppc
        max_enpaths = lambda x: min(self.ppc, len(x))
        # Saving path origins and directionality
        path_ori = [0] * (max_enpaths(in_hpaths) + max_enpaths(out_hpaths)) + \
                   [1] * (max_enpaths(in_tpaths) + max_enpaths(out_tpaths))
        path_dir = [0] * max_enpaths(in_hpaths) + [1] * max_enpaths(out_hpaths) + \
                   [0] * max_enpaths(in_tpaths) + [1] * max_enpaths(out_tpaths)

        # Sample paths for each entity context using local seed
        with du.local_seed(self.seed, self.epoch, index):
            context_paths = []  # holding all the context paths
            for context in [in_hpaths, out_hpaths, in_tpaths, out_tpaths]:
                context_paths += [context[i] for i in
                          torch.randperm(len(context))[:self.ppc]]

        ent_paths, rel_paths, head_idxs, tail_idxs, pos = [], [], [], [], []
        for i, context_path in enumerate(context_paths):
            sel = max if path_dir[i] == 0 else min  # for self-loop
            # Instantiate context paths to retrieve entities and relations
            ents, rels = self.fetch_path_context(*context_path)
            heix = sel((ents == head).nonzero()).item() if head in ents else -1
            teix = sel((ents == tail).nonzero()).item() if tail in ents else -1
            # Creates a custom positional encoding centered around the entity
            entix = heix if path_ori[i] == 0 else teix
            pos.append(plib.get_entfocused_positionals(
                entix*2, len(ents)*2-1) + 1)  # adding 1 to avoid padding
            # FIXME Mask the relation to predict if the triple occurs in paths
            # Encode both relations and entities, and mask the bridge relation
            rels = self.encode_relations(rels)
            # rels[head_idx] = self.tokens_to_idxs["MSK"]
            rel_paths.append(rels)
            ent_paths.append(self.encode_entities(ents))
            head_idxs.append(heix)
            tail_idxs.append(teix)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": head_idxs, "tail_indexes": tail_idxs,
            "relation": self.tokens_to_idxs[rel.item()],
            "pos": pos, "ori_triple": triple, "path_origins": path_ori,
        }

    def _getitem_combined(self, index) -> dict:
        """
        Get a multi-path contextualisation of the i-th triple in the dataset.
        Paths are constructed using both incoming and outgoing context.
        """
        triple = torch.as_tensor(self.triplestore[index])
        head, rel, tail = triple
        # Retrieving combined incoming and outgoing paths per entity
        with du.local_seed(self.seed, self.epoch, index):
            h_epaths, h_rpaths, h_idxs, h_erpos = \
                self._create_inout_contextpaths(head)
            t_epaths, t_rpaths, t_idxs, t_erpos = \
                self._create_inout_contextpaths(tail)
        no_hpaths, no_tpaths = len(h_epaths), len(t_epaths)
        # Record the origin of each path (0 for H, 1 for T) and update entixs
        path_ori = [0] * no_hpaths + [1] * no_tpaths
        h_idxs = h_idxs + [-1] * no_tpaths
        t_idxs = [-1] * no_hpaths + t_idxs

        return {
            "id": index,
            "pos": h_erpos + t_erpos, 
            "ent_paths": h_epaths + t_epaths,
            "rel_paths": h_rpaths + t_rpaths,
            "head_indexes": h_idxs, "tail_indexes": t_idxs,
            "relation": self.tokens_to_idxs[rel.item()],
            "ori_triple": triple, "path_origins": path_ori,
        }

    def __getitem__(self, index) -> dict:
        """
        Get a multi-paths contextualisation of the i-th triple in the dataset.
        """
        # self._getitem_separate(index)
        return self._getitem_combined(index)


class EntityMultiPathDataset(MultiPathDataset):

    def __init__(self,
                 path_store: str,
                 relcontext_store: str,
                 triple_store: torch.tensor,
                 unique_entities: torch.tensor,
                 paths_per_entity = 20,
                 seed: int = 46,
                ):
        """
        An entity-centric dataset where entities are defined from their paths.
        """
        super().__init__(path_store, relcontext_store, triple_store,
                         context_triple_store=triple_store,
                         maximum_triple_paths=paths_per_entity*2,
                         tokens_to_idxs=None, num_negatives=0,
                         triple_corruptor=None, seed=seed)
        # Recording path budget and spreading it across the in-out context
        self.unique_entities = unique_entities

    def __len__(self):
        return len(self.unique_entities)

    def _getitem_separate(self, index: int) -> dict:
        """
        Get a multi-path contextualisation of the i-th entity in the dataset.
        """
        ent = self.unique_entities[index]

        # Retrieving incoming and outgoing paths/triples per entity
        in_paths, out_paths = plib.create_contextpaths(
            ent, self.path_index, self.context_triple_store)
        # Record the origin of each path (0 for H, 1 for T) capped at ppc
        max_enpaths = lambda x: min(self.ppc, len(x))
        # Saving path directionality: 0 incoming, 1 outgoing
        path_dir = [0] * max_enpaths(in_paths) + [1] * max_enpaths(out_paths)

        # Sample paths for each entity context using local seed
        with du.local_seed(self.seed, index):
            context_paths = []  # holding all the context paths
            for context in [in_paths, out_paths]:
                context_paths += [context[i] for i in
                          torch.randperm(len(context))[:self.ppc]]

        ent_paths, rel_paths, ent_idxs, pos = [], [], [], []
        for i, context_path in enumerate(context_paths):
            sel = max if path_dir[i] == 0 else min  # for self-loop
            # Instantiate context paths to retrieve entities and relations
            ents, rels = self.fetch_path_context(*context_path)
            entix = sel((ents == ent).nonzero()).item() if ent in ents else -1
            # Creates a custom positional encoding centered around the entity
            pos.append(plib.get_entfocused_positionals(
                entix*2, len(ents)*2-1) + 1)  # adding 1 to avoid padding
            # Encode both relations and entities, and mask the bridge relation
            rels = self.encode_relations(rels)
            # rels[head_idx] = self.tokens_to_idxs["MSK"]
            rel_paths.append(rels)
            ent_paths.append(self.encode_entities(ents))
            ent_idxs.append(entix)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": ent_idxs, "pos": pos, 
        }

    def _getitem_combined(self, index) -> dict:
        """
        Get a multi-path contextualisation of the i-th entity in the dataset.
        """
        ent = self.unique_entities[index]
        # Retrieving combined incoming and outgoing paths per entity
        with du.local_seed(self.seed, index):
            ent_paths, rel_paths, ent_idxs, pos = \
                self._create_inout_contextpaths(ent)

        return {
            "id": index, "ent_paths": ent_paths, "rel_paths": rel_paths,
            "head_indexes": ent_idxs, "pos": pos, 
        }

    def __getitem__(self, index) -> dict:
        # return self._getitem_separate(index)
        return self._getitem_combined(index)


