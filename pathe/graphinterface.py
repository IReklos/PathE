import copy
import os

import igraph as ig
from kgloader import KgLoader
import torch
import statics
from typing import Dict, List
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import random
from itertools import chain, zip_longest


def get_batch_of_walks(graph, entities: torch.Tensor, num_paths: int = 10,
                       num_steps: int = 15,
                       unique_criterion: str = 'relation',
                       mixing_ratio: float = 1.0, max_attempts: int = 30):
    """
        A function that produces random walk paths starting from each entity
        of the KG. To be used for generating paths.
        :return: A Dict[node id:
        List[List[relation paths], List[entity paths], List[joined entity and
        relation paths]]. @param num_paths:The number of random walks to
        generate (default 10) @type num_paths: int
        """
    max_attempts = max_attempts
    if num_paths < 1:
        raise Exception("k must be at least 1.")
    vocab = {}
    if entities is None:
        raise Exception("The dataset does not contain entities.")

    nodes = entities.tolist()

    # single-threaded mining is found to be as fast as multi-processing +
    # igraph for some reason, so let's use a dummy for-loop
    total_paths = 0
    total_nodes = 0
    for i in tqdm(nodes):
        relation_paths = []
        entity_paths = []
        joined_paths = []
        unique_paths = set()
        remaining = num_paths
        max_retries = max_attempts
        # if i in self.dataloader.entity_stats[part]['tail_only']:
        #     continue
        try:
            while remaining > 0:
                if max_retries == 0:
                    break
                if random.random() < mixing_ratio:
                    path = graph.random_walk(start=i, steps=num_steps,
                                             return_type="edges",
                                             mode='out')
                else:
                    path = graph.random_walk(start=i, steps=num_steps,
                                             return_type="edges",
                                             mode='in')
                    path = path[::-1]
                max_retries -= 1

                if len(path) > 0:
                    e_path = [[graph.es[k].source for k in path] +
                              [graph.es[path[-1]].target]]
                    set_path = set(e_path[0])
                    if len(set_path) == len(e_path[0]) and len(
                            e_path[0]) > 2:
                        if unique_criterion == 'relation':
                            r_path = [
                                [graph.es[k]['edge_type'] for k in path]]
                            if str(r_path[0]) not in unique_paths:
                                relation_paths.append(r_path[0])
                                unique_paths.add(str(r_path[0]))
                                entity_paths.append(e_path[0])
                                joined_paths.append(
                                    [[x for x in chain.from_iterable(
                                        zip_longest(e_path[0],
                                                    r_path[0]))
                                      if x is not None]][0])
                                remaining -= 1
                        elif unique_criterion == 'entity':
                            if str(e_path[0]) not in unique_paths:
                                r_path = [
                                    [graph.es[k]['edge_type'] for k in
                                     path]][0]
                                relation_paths.append(r_path[0])
                                unique_paths.add(str(e_path[0]))
                                entity_paths.append(e_path[0])
                                joined_paths.append(
                                    [[x for x in chain.from_iterable(
                                        zip_longest(e_path[0],
                                                    r_path[0]))
                                      if x is not None]][0])
                                remaining -= 1
                        elif unique_criterion == 'joined':
                            r_path = [
                                [graph.es[k]['edge_type'] for k in path]][0]
                            j_path = [[x for x in chain.from_iterable(
                                zip_longest(e_path[0],
                                            r_path[0]))
                                       if x is not None]]
                            if str(j_path[0]) not in unique_paths:
                                relation_paths.append(r_path[0])
                                unique_paths.add(str(j_path[0]))
                                entity_paths.append(e_path[0])
                                joined_paths.append(j_path[0])
                                remaining -= 1
                else:
                    continue
        except:
            # continue
            print("Node " + str(i) + " not in the Graph")
        if len(relation_paths) > 0 and len(
                max(relation_paths, key=len)) > 1:
            vocab[i] = [relation_paths, entity_paths, joined_paths]
            total_nodes += 1
            total_paths += len(relation_paths)
        else:
            relation_paths = []
            entity_paths = []
            joined_paths = []
            unique_paths = set()
            remaining = num_paths
            max_retries = max_attempts
            try:
                while remaining > 0:
                    if max_retries == 0:
                        break
                    if random.random() < 1 - mixing_ratio:
                        path = graph.random_walk(start=i, steps=num_steps,
                                                 return_type="edges",
                                                 mode='out')
                    else:
                        path = graph.random_walk(start=i, steps=num_steps,
                                                 return_type="edges",
                                                 mode='in')
                        path = path[::-1]
                    max_retries -= 1

                    if len(path) > 0:
                        e_path = [[graph.es[k].source for k in path] +
                                  [graph.es[path[-1]].target]]
                        set_path = set(e_path[0])
                        if len(set_path) == len(e_path[0]) and len(
                                e_path[0]) > 2:
                            if unique_criterion == 'relation':
                                r_path = [
                                    [graph.es[k]['edge_type'] for k in path]]
                                if str(r_path[0]) not in unique_paths:
                                    relation_paths.append(r_path[0])
                                    unique_paths.add(str(r_path[0]))
                                    entity_paths.append(e_path[0])
                                    joined_paths.append(
                                        [[x for x in chain.from_iterable(
                                            zip_longest(e_path[0],
                                                        r_path[0]))
                                          if x is not None]][0])
                                    remaining -= 1
                            elif unique_criterion == 'entity':
                                if str(e_path[0]) not in unique_paths:
                                    r_path = [
                                        [graph.es[k]['edge_type'] for k in
                                         path]][0]
                                    relation_paths.append(r_path[0])
                                    unique_paths.add(str(e_path[0]))
                                    entity_paths.append(e_path[0])
                                    joined_paths.append(
                                        [[x for x in chain.from_iterable(
                                            zip_longest(e_path[0],
                                                        r_path[0]))
                                          if x is not None]][0])
                                    remaining -= 1
                            elif unique_criterion == 'joined':
                                r_path = [
                                    [graph.es[k]['edge_type'] for k in
                                     path]][0]
                                j_path = [[x for x in chain.from_iterable(
                                    zip_longest(e_path[0],
                                                r_path[0]))
                                           if x is not None]]
                                if str(j_path[0]) not in unique_paths:
                                    relation_paths.append(r_path[0])
                                    unique_paths.add(str(j_path[0]))
                                    entity_paths.append(e_path[0])
                                    joined_paths.append(j_path[0])
                                    remaining -= 1
                    else:
                        continue
            except:
                # continue
                print("Node " + str(i) + " not in graph")
            if len(relation_paths) > 0 and len(
                    max(relation_paths, key=len)) > 1:
                vocab[i] = [relation_paths, entity_paths, joined_paths]
                total_nodes += 1
                total_paths += len(relation_paths)
    # print("Total paths: " + str(total_paths))
    # print("Total nodes with paths: " + str(total_nodes))
    # print(
    #     "Average paths per node: " + str(float(total_paths) / total_nodes))
    # print("Nodes without paths: " + str(len(nodes) - total_nodes))

    return vocab


class Graph():
    """
    A class that handles the Graph and the anchor selection / path calculation.
    Inherits the igraph.Graph class so see documentation of iGraph for added
    functionality.
    """

    def __init__(self, dataloader: KgLoader, part: str = 'train'):
        """
        The constructor of the subclass
        :param dataloader: The dataloader of the graph triples
        :param part: The part of the triples to use for constructing the graph.
        Valid options are ['train', 'test', 'all']
        """
        self.dataloader = dataloader
        if self.dataloader.num_nodes_total <= 0:
            raise Exception("The Dataloader object cannot be empty.")
        self.part = part
        if self.part not in ['train', 'validation', 'test', 'all']:
            raise Exception("A valid part of the dataset must be selected to "
                            "create the Graph.")
        if type(self.dataloader.train) != torch.Tensor or \
                type(self.dataloader.validation) != torch.Tensor \
                or type(self.dataloader.test) != torch.Tensor:
            raise Exception("The triple data must be in torch.Tensor format.")
        self.num_nodes, self.edges, self.relations = self.get_super_init_params()
        # super().__init__(n=self.num_nodes, edges=self.edges,
        #                  edge_attrs={'edge_type': list(self.relations)},
        #                  directed=True)
        self.graph = ig.Graph(n=self.num_nodes, edges=self.edges,
                              edge_attrs={'edge_type': list(self.relations)},
                              directed=True)

    def get_super_init_params(self):
        """
        Get the initialization parameters for the igGraph Graph super class
        depending on the part of the dataset specified. :return: Triple (
        number of unique nodes, edges List[Tuple(start node, end node)],
        relation types for each edge in numpy vector)
        """
        if self.part == 'train':  # XXX to read as development set (train +
            # valid)
            triples = self.dataloader.train
        elif self.part == 'validation':
            triples = self.dataloader.validation
        elif self.part == 'test':
            triples = self.dataloader.test
        else:
            triples = torch.cat((self.dataloader.train,
                                 self.dataloader.validation,
                                 self.dataloader.test), dim=0)
        heads, relations, tails = triples[:, 0].numpy(), triples[:, 1].numpy(), \
            triples[:, 2].numpy()
        edges = [[h, t] for h, t in zip(heads, tails)]
        return statics.count_unique_nodes(triples), edges, relations

    def find_anchors(self, anchor_strategy: Dict[str, float], num_anchors):
        """
        Nodepiece anchor selection. :param anchor_strategy: A dictionary of {
        strategy:percentage} pairs. Valid strategies include ["degree",
        "pagerank", "random"] and the percentages must sum to 1 :param
        num_anchors: The number of anchors to be selected :return: A list of
        anchors
        """
        if sum(anchor_strategy.values()) > 1.0:
            raise Exception("The sum of the strategy percentages must not "
                            "exceed 1.0")
        # sampling anchor nodes
        anchors = []
        for strategy, ratio in anchor_strategy.items():
            if ratio <= 0.0:
                continue
            topK = int(np.ceil(ratio * num_anchors))
            print(f"Computing the {strategy} nodes")
            if strategy == "degree":
                top_nodes = sorted(
                    [(i, n) for i, n in enumerate(self.graph.degree())],
                    key=lambda x: x[1], reverse=True)
            elif strategy == "betweenness":
                # This is O(V^3) - disabled
                raise NotImplementedError("Betweenness is disabled due to "
                                          "computational costs")
            elif strategy == "pagerank":
                top_nodes = sorted([(i, n) for i, n in
                                    enumerate(
                                        self.graph.personalized_pagerank())],
                                   key=lambda x: x[1],
                                   reverse=True)
            elif strategy == "random":
                top_nodes = [(int(k), 1) for k in
                             np.random.permutation(
                                 np.arange(self.dataloader.num_nodes_total))]

            selected_nodes = [node for node, d in top_nodes if
                              node not in anchors][:topK]

            anchors.extend(selected_nodes)
            print(
                f"Added {len(selected_nodes)} nodes under the {strategy} strategy")

        return anchors

    def find_highest_outdegree_anchors(self, num_anchors: int):
        """
        A function that selects anchors based on the intuition that anchors
        should be selected from the head entities with the highest number of
        outgoing relations. Based on this there will be many paths starting
        from these entities and should provide good reachability. :param
        num_anchors: The number of anchors to select :type num_anchors: int
        :return: The anchors :rtype: List[int]
        """
        triples = self.dataloader.train_no_inv
        unique, counts = torch.unique(triples[:, 0], return_counts=True)
        heads_counts = torch.column_stack((unique, counts))
        sorting_indices = heads_counts[:, -1].argsort(dim=0, descending=True)
        sorted_heads = unique[sorting_indices]
        anchors = sorted_heads[:num_anchors]
        return anchors.tolist()

    def get_unreachable_nodes(self):
        """
        Find the entities that exist only as heads of triples and
        as such are unreachable without adding inverse edges.
        :return: The unreachable nodes
        :rtype: List[int]
        """
        unreachable_nodes = self.dataloader.entity_stats[self.part]['head_only']
        return unreachable_nodes

    def get_all_paths(self, anchors: List[int], mode: str = "path",
                      path_type: str = "shortest",
                      path_num_limit: int = 0):
        """
        A function that produces at most path_num_limit shortest paths from
        each node to anchors. If fewer than path_num_limit paths are found (
        or no paths in case of a disconnected node) the missing paths are
        denoted using the NOTHING_TOKEN (-99). Each path is a List containing
        the anchor node at the start of the path and the edges traversed to
        get to the target node. :param anchors: The set of anchor node ids
        :param mode: Path or breadth first search (currently only path is
        supported) :param path_type: The type of paths to return. Either
        shortest paths, randomly selected paths or all the paths. :param
        path_num_limit: The number of paths to find, If set to 0 finds
        num_anchors paths :return: A Dict[node id: List[paths]].
        @param path_num_limit:
        @type path_num_limit:
        @param mode:
        @type mode:
        @param anchors:
        @type anchors:
        @param path_type:
        @type path_type:
        """
        if path_type not in ["shortest", "random", "all"]:
            raise Exception("path_type must be one of shortest, random, all.")
        if path_num_limit == 0:
            limit = len(anchors)
        else:
            limit = path_num_limit
        vocab = {}
        # if path_type == "shortest": print(f"Computing the entity vocabulary
        # - paths, retaining {limit} shortest paths per node") else: print(
        # f"Computing the entity vocabulary - paths, retaining {limit} random
        # paths per node")

        # only required for BFS search
        # anc_set = set(anchors)

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        for i in tqdm(range(self.dataloader.num_nodes_total)):
            if mode == "path":
                paths = self.graph.get_shortest_paths(v=i, to=anchors,
                                                      output="epath",
                                                      mode='in')
                if len(paths) > 0 and len(max(paths, key=len)) > 0:
                    # graph.es[path[-1]].source means that in the edge
                    # sequence of the graph, get the edge in the end of the
                    # path (because of the mode= 'in', get_shortest_paths
                    # returns the edges of the path in reverse order (target
                    # node (anchors in our case) ->start node)) and because
                    # of the epath as output, the function returns the path
                    # as an edge sequence. so we get the first edge of the
                    # path from the anchor and get the source (the anchor).
                    # path[::-1] reverses the list
                    relation_paths = [[self.graph.es[path[-1]].source] +
                                      [self.graph.es[k]['edge_type'] for k in
                                       path[::-1]]
                                      for path in paths if len(path) > 0]
                else:
                    # if NO anchor can be reached from the node - encode with
                    # a special NOTHING_TOKEN
                    relation_paths = [[statics.NOTHING_TOKEN] for _ in
                                      range(limit)]
                if path_type == "shortest":
                    relation_paths = sorted(relation_paths,
                                            key=lambda x: len(x))[:limit]
                if path_type == "random":
                    random.shuffle(relation_paths)
                    relation_paths = relation_paths[:limit]
                vocab[i] = relation_paths
            else:
                raise NotImplementedError

        return vocab

    def get_shortest_k_paths_between(self, triples, mode: str = "path",
                                     num_paths: int = 1):
        """
        A function that produces k shortest paths from each entity to every
        other entity of the KG. To be used for generating paths between nodes
        not between nodes and anchors :param mode: Path or breadth first
        search (currently only path is supported) :return: A Dict[node id:
        List[List[relation paths], List[entity paths], List[joined entity and
        relation paths]]. @param num_paths:The number of shortest paths to
        generate (default 1) @type num_paths: int
        """
        if num_paths < 1:
            raise Exception("k must be at least 1.")
        vocab = {}
        # for stats
        total_head_paths = 0
        total_tail_paths = 0
        total_triples = 0
        triples_without_head_paths = 0
        triples_without_tail_paths = 0

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        for i in tqdm(range(triples.size()[0])):
            total_triples += 1
            if mode == "path":
                head_paths = []
                tail_paths = []

                visited_head = set()
                visited_tail = set()
                head_node_paths = self.graph.get_k_shortest_paths(
                    v=triples[i, 0],
                    to=triples[i, 2],
                    k=num_paths,
                    output="epath",
                    mode='out')
                tail_node_paths = self.graph.get_k_shortest_paths(
                    v=triples[i, 2],
                    to=triples[i, 0],
                    k=num_paths,
                    output="epath",
                    mode='out')
                # ensure that the paths between each (star,end) node are
                # unique
                for candidate in head_node_paths:
                    # convert the path to string of edge ids
                    str_candidate = str([self.graph.es[k]['edge_type']
                                         for k in candidate])
                    # check if path has been visited
                    if str_candidate not in visited_head:
                        # if not add to visited and to paths
                        head_paths.extend([candidate])
                        visited_head.add(str_candidate)
                if len(head_paths) > 0 and len(max(head_paths, key=len)) > 0:
                    total_head_paths += len(head_paths)
                    # in the edge sequence of the graph get for each edge in
                    # each path the edge type and add it to the relational
                    # path list
                    relation_paths = [
                        [self.graph.es[k]['edge_type'] for k in path]
                        for path in head_paths if len(path) > 1]
                    # in the edge sequence of the graph get for each edge in
                    # each path the source node and add it to the list.
                    # Finally, concatenate the target of the last edge to get
                    # the last (target) node in the path.
                    entity_paths = [[self.graph.es[k].source for k in path] +
                                    [self.graph.es[path[-1]].target]
                                    for path in head_paths if len(path) > 1]
                else:
                    # if NO node can be reached from the source node skip it
                    triples_without_head_paths += 1
                    continue
                vocab[str(triples[i, 0].item()) + ";" + str(triples[i, 2].item(

                ))] = [
                    relation_paths,
                    entity_paths]
                for candidate in tail_node_paths:
                    # convert the path to string of edge ids
                    str_candidate = str([self.graph.es[k]['edge_type']
                                         for k in candidate])
                    # check if path has been visited
                    if str_candidate not in visited_tail:
                        # if not add to visited and to paths
                        tail_paths.extend([candidate])
                        visited_tail.add(str_candidate)
                if len(tail_paths) > 0 and len(max(tail_paths, key=len)) > 0:
                    total_tail_paths += 1
                    # in the edge sequence of the graph get for each edge in
                    # each path the edge type and add it to the relational
                    # path list
                    relation_paths = [
                        [self.graph.es[k]['edge_type'] for k in path]
                        for path in tail_paths if len(path) > 1]
                    # in the edge sequence of the graph get for each edge in
                    # each path the source node and add it to the list.
                    # Finally, concatenate the target of the last edge to get
                    # the last (target) node in the path.
                    entity_paths = [[self.graph.es[k].source for k in path] +
                                    [self.graph.es[path[-1]].target]
                                    for path in tail_paths if len(path) > 1]
                else:
                    # if NO node can be reached from the source node skip it
                    triples_without_tail_paths += 1
                    continue
                vocab[str(triples[i, 2].item()) + ";" + str(triples[i, 0].item(

                ))] = [
                    relation_paths,
                    entity_paths]
            else:
                raise NotImplementedError
        print("STATS:")
        print("total paths form head nodes: ", total_head_paths)
        print("total paths form tail nodes: ", total_tail_paths)
        print("Average head paths per triple: ",
              total_head_paths / float(total_triples))
        print("Average tail paths per triple: ",
              total_tail_paths / float(total_triples))
        print("Triples with no head->tail paths: ", triples_without_head_paths)
        print("triples without tail->head paths: ", triples_without_tail_paths)

        return vocab

    def get_shortest_k_paths(self, mode: str = "path", num_paths: int = 1):
        """
        A function that produces k shortest paths from each entity to every
        other entity of the KG. To be used for generating paths between nodes
        not between nodes and anchors :param mode: Path or breadth first
        search (currently only path is supported) :return: A Dict[node id:
        List[List[relation paths], List[entity paths], List[joined entity and
        relation paths]]. @param num_paths:The number of shortest paths to
        generate (default 1) @type num_paths: int
        """
        if num_paths < 1:
            raise Exception("k must be at least 1.")
        vocab = {}
        if self.dataloader.unique_entities is None:
            raise Exception("The dataset does not contain entities.")
        nodes = self.dataloader.unique_entities.tolist()

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        for i in tqdm(nodes):
            if mode == "path":
                paths = []
                for node in nodes:
                    visited = set()
                    node_paths = self.graph.get_k_shortest_paths(v=i, to=node,
                                                                 k=num_paths,
                                                                 output="epath",
                                                                 mode='out')
                    # ensure that the paths between each (star,end) node are
                    # unique
                    for candidate in node_paths:
                        # convert the path to string of edge ids
                        str_candidate = str([self.graph.es[k]['edge_type']
                                             for k in candidate])
                        # check if path has been visited
                        if str_candidate not in visited:
                            # if not add to visited and to paths
                            paths.extend([candidate])
                            visited.add(str_candidate)
                if len(paths) > 0 and len(max(paths, key=len)) > 0:
                    # in the edge sequence of the graph get for each edge in
                    # each path the edge type and add it to the relational
                    # path list
                    relation_paths = [
                        [self.graph.es[k]['edge_type'] for k in path]
                        for path in paths if len(path) > 1]
                    # in the edge sequence of the graph get for each edge in
                    # each path the source node and add it to the list.
                    # Finally, concatenate the target of the last edge to get
                    # the last (target) node in the path.
                    entity_paths = [[self.graph.es[k].source for k in path] +
                                    [self.graph.es[path[-1]].target]
                                    for path in paths if len(path) > 1]
                    # Some python magic to interleave the above two lists
                    joined_paths = [[x for x in chain.from_iterable(
                        zip_longest(entity_paths[i], relation_paths[i]))
                                     if x is not None] for i in
                                    range(len(relation_paths))]
                else:
                    # if NO node can be reached from the source node skip it
                    continue
                vocab[i] = [relation_paths, entity_paths, joined_paths]
            else:
                raise NotImplementedError

        return vocab

    def get_shortest_paths_to_nodes(self, nodes: List = None, mode: str = \
            "path", path_type: str = "shortest"):
        """
        A function that produces all the shortest paths from each entity to
        every other entity of the KG. To be used for generating paths between
        nodes not between nodes and anchors :param mode: Path or breadth
        first search (currently only path is supported) :param path_type: The
        type of paths to return. Either shortest paths, randomly selected
        paths or all the paths. :return: A Dict[node id: List[List[relation
        paths], List[entity paths], List[joined entity and relation paths]].
        """
        if path_type not in ["shortest", "random", "all"]:
            raise Exception("path_type must be one of shortest, random, all.")
        vocab = {}
        if self.dataloader.unique_entities is None:
            raise Exception("The dataset does not contain entities.")
        if not nodes:
            nodes = self.dataloader.unique_entities.tolist()
        start_nodes = self.dataloader.unique_entities.tolist()

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        for i in tqdm(start_nodes):
            if mode == "path":
                paths = self.graph.get_shortest_paths(v=i, to=nodes,
                                                      output="epath",
                                                      mode='out')
                if len(max(paths, key=len)) > 0:
                    # in the edge sequence of the graph get for each edge in
                    # each path the edge type and add it to the relational
                    # path list
                    relation_paths = [
                        [self.graph.es[k]['edge_type'] for k in path]
                        for path in paths if len(path) > 0]
                    # in the edge sequence of the graph get for each edge in
                    # each path the source node and add it to the list.
                    # Finally, concatenate the target of the last edge to get
                    # the last (target) node in the path.
                    entity_paths = [[self.graph.es[k].source for k in path] + [
                        self.graph.es[path[-1]].target]
                                    for path in paths if len(path) > 0]
                    # Some python magic to interleave the above two lists
                    joined_paths = [[x for x in chain.from_iterable(
                        zip_longest(entity_paths[i], relation_paths[i]))
                                     if x is not None] for i in
                                    range(len(relation_paths))]
                else:
                    # if NO node can be reached from the source node skip it
                    continue
                vocab[i] = [relation_paths, entity_paths, joined_paths]
            else:
                raise NotImplementedError

        return vocab

    def get_simple_paths_between_nodes(self, start_nodes: List = None,
                                       target_nodes: List = None,
                                       mode: str = "path",
                                       path_type: str = "shortest"):
        """
        A function that produces all the shortest paths from each entity to
        every other entity of the KG. To be used for generating paths between
        nodes not between nodes and anchors :param mode: Path or breadth
        first search (currently only path is supported) :param path_type: The
        type of paths to return. Either shortest paths, randomly selected
        paths or all the paths. :return: A Dict[node id: List[List[relation
        paths], List[entity paths], List[joined entity and relation paths]].
        """
        vocab = {}
        if self.dataloader.unique_entities is None:
            raise Exception("The dataset does not contain entities.")
        if not target_nodes:
            target_nodes = self.dataloader.unique_entities.tolist()
        if not start_nodes:
            start_nodes = self.dataloader.unique_entities.tolist()

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        for i in tqdm(start_nodes):
            if mode == "path":
                paths = self.graph.get_all_simple_paths(v=i, to=target_nodes,
                                                        cutoff=20,
                                                        mode='out')
                if len(paths) > 0 and len(max(paths, key=len)) > 0:
                    # in the edge sequence of the graph get for each edge in
                    # each path the edge type and add it to the relational
                    # path list
                    relation_paths = [
                        [self.graph.es[k]['edge_type'] for k in path]
                        for path in paths if len(path) > 0]
                    # in the edge sequence of the graph get for each edge in
                    # each path the source node and add it to the list.
                    # Finally, concatenate the target of the last edge to get
                    # the last (target) node in the path.
                    entity_paths = [[self.graph.es[k].source for k in path] + [
                        self.graph.es[path[-1]].target]
                                    for path in paths if len(path) > 0]
                    # Some python magic to interleave the above two lists
                    joined_paths = [[x for x in chain.from_iterable(
                        zip_longest(entity_paths[i], relation_paths[i]))
                                     if x is not None] for i in
                                    range(len(relation_paths))]
                else:
                    # if NO node can be reached from the source node skip it
                    continue
                vocab[i] = [relation_paths, entity_paths, joined_paths]
            else:
                raise NotImplementedError

        return vocab

    def get_parallel_random_walks(self, num_paths: int = 10,
                         num_steps: int = 15,
                         unique_criterion: str = 'relation',
                         part: str = 'train', mixing_ratio: float = 1.0,
                                  max_attempts: int = 30):
        """
        A function that produces random walk paths starting from each entity
        of the KG using parallel mining.
        :return: A Dict[node id:
        List[List[relation paths], List[entity paths], List[joined entity and
        relation paths]].
        @param num_paths:The number of random walks to
        generate (default 10)
        @type num_paths: int

        """

        if num_paths < 1:
            raise Exception("k must be at least 1.")
        vocab = {}
        if self.dataloader.unique_entities is None:
            raise Exception("The dataset does not contain entities.")
        if self.part == 'train':
            nodes = self.dataloader.unique_entities
        elif self.part == 'validation':
            nodes = self.dataloader.unique_val
        else:
            nodes = self.dataloader.unique_test

        worker_count = os.cpu_count() // 2
        slice_size = nodes.size()[0] // worker_count

        walks_list = Parallel(n_jobs=worker_count, backend="loky") \
            (delayed(get_batch_of_walks)(self.graph,
                                         nodes[i:i + slice_size],
                                         num_paths, num_steps, unique_criterion,
                                         mixing_ratio, max_attempts) for i
             in tqdm(range(0, nodes.size()[0], slice_size)))

        for walk_dict in walks_list:
            vocab.update(walk_dict)

        return vocab

    def get_random_walks(self, num_paths: int = 10,
                         num_steps: int = 15,
                         unique_criterion: str = 'relation',
                         part: str = 'train', mixing_ratio: float = 1.0,
                         max_attempts: int = 30):
        """
        A function that produces random walk paths starting from each entity
        of the KG. To be used for generating paths
        :return: A Dict[node id:
        List[List[relation paths], List[entity paths], List[joined entity and
        relation paths]]. @param num_paths:The number of random walks to
        generate (default 10) @type num_paths: int
        """
        max_attempts = max_attempts
        if num_paths < 1:
            raise Exception("k must be at least 1.")
        vocab = {}
        if self.dataloader.unique_entities is None:
            raise Exception("The dataset does not contain entities.")
        if self.part == 'train':
            nodes = self.dataloader.unique_entities.tolist()
        elif self.part == 'validation':
            nodes = self.dataloader.unique_val.tolist()
        else:
            nodes = self.dataloader.unique_test.tolist()

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        total_paths = 0
        total_nodes = 0
        for i in tqdm(nodes):
            relation_paths = []
            entity_paths = []
            joined_paths = []
            unique_paths = set()
            remaining = num_paths
            max_retries = max_attempts
            # if i in self.dataloader.entity_stats[part]['tail_only']:
            #     continue
            try:
                while remaining > 0:
                    if max_retries == 0:
                        break
                    if random.random() < mixing_ratio:
                        path = self.graph.random_walk(start=i, steps=num_steps,
                                                return_type="edges",
                                                mode='out')
                    else:
                        path = self.graph.random_walk(start=i, steps=num_steps,
                                                return_type="edges",
                                                mode='in')
                        path = path[::-1]
                    max_retries -= 1

                    if len(path) > 0:
                        e_path = [[self.graph.es[k].source for k in path] +
                                  [self.graph.es[path[-1]].target]]
                        set_path = set(e_path[0])
                        if len(set_path) == len(e_path[0]) and len(
                                e_path[0]) > 2:
                            if unique_criterion == 'relation':
                                r_path = [
                                    [self.graph.es[k]['edge_type'] for k in
                                     path]]
                                if str(r_path[0]) not in unique_paths:
                                    relation_paths.append(r_path[0])
                                    unique_paths.add(str(r_path[0]))
                                    entity_paths.append(e_path[0])
                                    joined_paths.append(
                                        [[x for x in chain.from_iterable(
                                            zip_longest(e_path[0],
                                                        r_path[0]))
                                          if x is not None]][0])
                                    remaining -= 1
                            elif unique_criterion == 'entity':
                                if str(e_path[0]) not in unique_paths:
                                    r_path = [
                                        [self.graph.es[k]['edge_type'] for k in
                                         path]][0]
                                    relation_paths.append(r_path[0])
                                    unique_paths.add(str(e_path[0]))
                                    entity_paths.append(e_path[0])
                                    joined_paths.append(
                                        [[x for x in chain.from_iterable(
                                            zip_longest(e_path[0],
                                                        r_path[0]))
                                          if x is not None]][0])
                                    remaining -= 1
                            elif unique_criterion == 'joined':
                                r_path = [
                                    [self.graph.es[k]['edge_type'] for k in path]][0]
                                j_path = [[x for x in chain.from_iterable(
                                    zip_longest(e_path[0],
                                                r_path[0]))
                                           if x is not None]]
                                if str(j_path[0]) not in unique_paths:
                                    relation_paths.append(r_path[0])
                                    unique_paths.add(str(j_path[0]))
                                    entity_paths.append(e_path[0])
                                    joined_paths.append(j_path[0])
                                    remaining -= 1
                    else:
                        continue
            except:
                print("Node " + str(i) + " not in the Graph")
            # if len(paths) > 0 and len(max(paths, key=len)) > 1:
            #     # in the edge sequence of the graph get for each edge in
            #     # each path the edge type and add it to the relational
            #     # path list
            #     relation_paths = [
            #         [self.es[k]['edge_type'] for k in path]
            #         for path in paths if len(path) > 1]
            #     # in the edge sequence of the graph get for each edge in
            #     # each path the source node and add it to the list.
            #     # Finally, concatenate the target of the last edge to get
            #     # the last (target) node in the path.
            #     entity_paths = [[self.es[k].source for k in path] +
            #                     [self.es[path[-1]].target]
            #                     for path in paths if len(path) > 1]
            #     # Some python magic to interleave the above two lists
            #     joined_paths = [[x for x in chain.from_iterable(
            #         zip_longest(entity_paths[i], relation_paths[i]))
            #                      if x is not None] for i in
            #                     range(len(relation_paths))]
            if len(relation_paths) > 0 and len(
                    max(relation_paths, key=len)) > 1:
                vocab[i] = [relation_paths, entity_paths, joined_paths]
                total_nodes += 1
                total_paths += len(relation_paths)
            else:
                relation_paths = []
                entity_paths = []
                joined_paths = []
                unique_paths = set()
                remaining = num_paths
                max_retries = max_attempts
                try:
                    while remaining > 0:
                        if max_retries == 0:
                            break
                        if random.random() < 1 - mixing_ratio:
                            path = self.graph.random_walk(start=i, steps=num_steps,
                                                    return_type="edges",
                                                    mode='out')
                        else:
                            path = self.graph.random_walk(start=i, steps=num_steps,
                                                    return_type="edges",
                                                    mode='in')
                            path = path[::-1]
                        max_retries -= 1

                        if len(path) > 0:
                            e_path = [[self.graph.es[k].source for k in path] +
                                      [self.graph.es[path[-1]].target]]
                            set_path = set(e_path[0])
                            if len(set_path) == len(e_path[0]) and len(
                                    e_path[0]) > 2:
                                if unique_criterion == 'relation':
                                    r_path = [
                                        [self.graph.es[k]['edge_type'] for k in path]]
                                    if str(r_path[0]) not in unique_paths:
                                        relation_paths.append(r_path[0])
                                        unique_paths.add(str(r_path[0]))
                                        entity_paths.append(e_path[0])
                                        joined_paths.append(
                                            [[x for x in chain.from_iterable(
                                                zip_longest(e_path[0],
                                                            r_path[0]))
                                              if x is not None]][0])
                                        remaining -= 1
                                elif unique_criterion == 'entity':
                                    if str(e_path[0]) not in unique_paths:
                                        r_path = [
                                            [self.graph.es[k]['edge_type'] for k in
                                             path]][0]
                                        relation_paths.append(r_path[0])
                                        unique_paths.add(str(e_path[0]))
                                        entity_paths.append(e_path[0])
                                        joined_paths.append(
                                            [[x for x in chain.from_iterable(
                                                zip_longest(e_path[0],
                                                            r_path[0]))
                                              if x is not None]][0])
                                        remaining -= 1
                                elif unique_criterion == 'joined':
                                    r_path = [
                                        [self.graph.es[k]['edge_type'] for k in
                                         path]][0]
                                    j_path = [[x for x in chain.from_iterable(
                                        zip_longest(e_path[0],
                                                    r_path[0]))
                                               if x is not None]]
                                    if str(j_path[0]) not in unique_paths:
                                        relation_paths.append(r_path[0])
                                        unique_paths.add(str(j_path[0]))
                                        entity_paths.append(e_path[0])
                                        joined_paths.append(j_path[0])
                                        remaining -= 1
                        else:
                            continue
                except:
                    print("Node " + str(i) + " not in graph")
                if len(relation_paths) > 0 and len(
                        max(relation_paths, key=len)) > 1:
                    vocab[i] = [relation_paths, entity_paths, joined_paths]
                    total_nodes += 1
                    total_paths += len(relation_paths)
        print("Total paths: " + str(total_paths))
        print("Total nodes with paths: " + str(total_nodes))
        print(
            "Average paths per node: " + str(float(total_paths) / total_nodes))
        print("Nodes without paths: " + str(len(nodes) - total_nodes))

        return vocab

    def get_paths(self, target_node, anchors: List[int], mode: str = "path",
                  path_type: str = "shortest",
                  path_num_limit: int = 0, exclude_self: bool = True):
        """
        A function that produces at most `path_num_limit` shortest paths from
        the target node to anchors. If fewer than `path_num_limit` paths are
        found (or no paths in case of a disconnected node) the missing paths
        are denoted using the NOTHING_TOKEN (-99). Each path is a List
        containing the anchor node at the start of the path and the edges
        traversed to get to the target node. :param target_node: The target
        node for the paths :param anchors: The set of anchor node ids :param
        mode: Path or breadth first search (currently only path is supported)
        :param path_type: The type of paths to return. Either shortest paths,
        randomly selected paths or all the paths. :param path_num_limit: The
        number of paths to find, If set to 0 finds num_anchors paths :param
        exclude_self: Whether to ignore the anchor itself when producing
        paths to anchor nodes :return: A Dict[node id: List[paths]].
        """
        anchor_set = set(anchors)
        if path_type not in ["shortest", "random", "all"]:
            raise Exception("path_type must be one of shortest, random, all.")
        if path_num_limit == 0:
            limit = len(anchors)
        else:
            limit = path_num_limit
        vocab = {}
        # if path_type == "shortest": print(f"Computing the entity vocabulary
        # - paths, retaining {limit} shortest paths per node") else: print(
        # f"Computing the entity vocabulary - paths, retaining {limit} random
        # paths per node")

        # Only required for BFS search
        # anc_set = set(anchors)

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        if mode == "path":
            if exclude_self and target_node in anchor_set:
                anchor_set.remove(target_node)
            paths = self.graph.get_shortest_paths(v=target_node, to=anchor_set,
                                                  output="epath", mode='in')
            if len(max(paths, key=len)) > 0:  # if at least one path was found
                # `graph.es[path[-1]].source` means that in the edge sequence
                # of the graph, get the edge in the end of the path (because
                # of the mode= 'in', get_shortest_paths returns the edges of
                # the path in reverse order (finish node->start node)) and
                # because of the epath as output, the function returns the
                # path as an edge sequence. So we get the first edge of the
                # path from the anchor and get the source (the anchor). path[
                # ::-1] reverses the list
                relation_paths = [[self.graph.es[path[-1]].source] +
                                  [self.graph.es[k]['edge_type'] for k in
                                   path[::-1]]
                                  for path in paths if len(path) > 0]
            else:
                # if NO anchor can be reached from the node - encode with a
                # special NOTHING_TOKEN
                relation_paths = [[statics.NOTHING_TOKEN] for _ in range(limit)]
            if path_type == "shortest":
                relation_paths = sorted(relation_paths, key=lambda x: len(x))[
                                 :limit]
            if path_type == "random":
                random.shuffle(relation_paths)
                relation_paths = relation_paths[:limit]
            vocab[target_node] = relation_paths
        return vocab

    def get_paths_with_indices(self, target_node, anchors: List[int],
                               mode: str = "path", path_type: str = "shortest",
                               path_num_limit: int = 0,
                               exclude_self: bool = True):
        """
        A function that produces at most `path_num_limit` shortest paths from
        the target node to anchors. If fewer than `path_num_limit` paths are
        found (or no paths in case of a disconnected node) the missing paths
        are denoted using the NOTHING_TOKEN (-99). Each path is a List
        containing the anchor node at the start of the path and the edges
        traversed to get to the target node. :param target_node: The target
        node for the paths :param anchors: The set of anchor node ids :param
        mode: Path or breadth first search (currently only path is supported)
        :param path_type: The type of paths to return. Either shortest paths,
        randomly selected paths or all the paths. :param path_num_limit: The
        number of paths to find, If set to 0 finds num_anchors paths :param
        exclude_self: Whether to ignore the anchor itself when producing
        paths to anchor nodes :return: A Dict[node id: List[paths]].
        """
        anchor_set = set(anchors)
        if path_type not in ["shortest", "random", "all"]:
            raise Exception("path_type must be one of shortest, random, all.")
        if path_num_limit == 0:
            limit = len(anchors)
        else:
            limit = path_num_limit
        vocab = {}
        # if path_type == "shortest": print(f"Computing the entity vocabulary
        # - paths, retaining {limit} shortest paths per node") else: print(
        # f"Computing the entity vocabulary - paths, retaining {limit} random
        # paths per node")

        # Only required for BFS search
        # anc_set = set(anchors)

        # single-threaded mining is found to be as fast as multi-processing +
        # igraph for some reason, so let's use a dummy for-loop
        if mode == "path":
            if exclude_self and target_node in anchor_set:
                anchor_set.remove(target_node)
            paths = self.graph.get_shortest_paths(v=target_node,
                                                  to=anchor_set, output="epath",
                                                  mode='in')
            if len(max(paths, key=len)) > 0:  # if at least one path was found
                # `graph.es[path[-1]].source` means that in the edge sequence
                # of the graph, get the edge in the end of the path (because
                # of the mode= 'in', get_shortest_paths returns the edges of
                # the path in reverse order (finish node->start node)) and
                # because of the epath as output, the function returns the
                # path as an edge sequence. So we get the first edge of the
                # path from the anchor and get the source (the anchor). path[
                # ::-1] reverses the list relation_paths = [[self.es[path[
                # -1]].source] + [self.es[k]['edge_type'] for k in path[
                # ::-1]] for path in paths if len(path) > 0] edge_list = [[
                # self.es[path[-1]].source] + [self.es[k].tuple for k in
                # path[::-1]] for path in paths if len(path) > 0]
                relation_paths = [
                    [self.graph.es[path[-1]].source] + [k for k in path[::-1]]
                    for path in paths if len(path) > 0]
            else:
                # if NO anchor can be reached from the node - encode with a
                # special NOTHING_TOKEN
                relation_paths = [[statics.NOTHING_TOKEN] for _ in range(limit)]
            if path_type == "shortest":
                relation_paths = sorted(relation_paths, key=lambda x: len(x))[
                                 :limit]
                # relation_paths = sorted(relation_paths, key=lambda x: len(
                # x))[:limit]
            if path_type == "random":
                random.shuffle(relation_paths)
                relation_paths = relation_paths[:limit]
            vocab[target_node] = relation_paths
        return vocab

    def get_relational_context(self):
        """
        A function that produces the incoming and outgoing edges for each node in the KG
        @return: a dictionary of lists of incoming and outgoing edges foe each node
        @rtype: Dict[int:Dict[str:List]]
        """
        vocab = {}
        if self.part == 'train':
            nodes = self.dataloader.unique_entities.tolist()
        elif self.part == 'validation':
            nodes = self.dataloader.unique_val.tolist()
        else:
            nodes = self.dataloader.unique_test.tolist()
        # for each node in the KG
        for i in tqdm(nodes):
            try:
                # get outgoing and incoming edges as edgesequence objects
                out_edges = self.graph.incident(vertex=i, mode='out')
                in_edges = self.graph.incident(vertex=i, mode='in')
                # covert them to lists of edge types
                out_edge_ids = [self.graph.es[k]['edge_type'] for k in out_edges
                                if
                                len(out_edges) > 0]
                in_edge_ids = [self.graph.es[k]['edge_type'] for k in in_edges
                               if
                               len(in_edges) > 0]
                # create a dictionary containing the incoming and outgoing edges
                # of the node
                node_dict = {}
                if len(out_edge_ids) > 0:
                    node_dict["out"] = out_edge_ids
                else:
                    node_dict["out"] = []
                if len(in_edge_ids) > 0:
                    node_dict["in"] = in_edge_ids
                else:
                    node_dict["in"] = []
                # aggregate relational contexts for all nodes in a vocab
                vocab[i] = node_dict
            except:
                print("Node " + str(i) + " not in graph.")
                continue
        return vocab

    @property
    def unique_relations(self):
        return torch.unique(torch.from_numpy(self.relations))

    @property
    def all_edges(self):
        return self.edges

    @property
    def node_count(self):
        return self.num_nodes
