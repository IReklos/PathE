import statics
from graphinterface import Graph
from kgloader import KgLoader
from typing import Iterable, Set
import copy
import os
import pickle


class Handler:
    """
    A class that creates the graph and handles node selection and path finding as well as node and edge mapping to
    consecutive ids for embedding.
    """

    # TODO path num and anchor num as percentage of nodes
    def __init__(self, dataloader: KgLoader, part: str = 'train',
                 strategy=None,
                 num_anchors: int = None, max_num_paths: int = None, use_cached: bool = True):
        """
        The class constructor
        :type strategy: Dict[str:float]
        :param dataloader: The dataloader of the graph triples
        :param part: The part of the triples to use for constructing the graph.
        Valid options are ['train', 'test', 'all']
        :param strategy: A dictionary of {strategy:percentage} pairs. Valid strategies include ["degree",
        "pagerank", "random"] and the percentages must sum to 1
        :param num_anchors: The number of anchors to be selected
        :param max_num_paths: The number of paths to find, If set to 0 finds num_anchors paths
        """
        self.part = part
        self.use_cached = use_cached
        self.dataloader = dataloader
        # Edge IDs start from the end of the node IDs (as offsets)
        self.edge_offset = self.dataloader.num_total_nodes
        self.graph = Graph(dataloader=self.dataloader, part=part)
        self.special_tokens = [statics.NOTHING_TOKEN, statics.CLS_TOKEN, statics.MASK_TOKEN,
                               statics.PADDING_TOKEN, statics.SEP_TOKEN]
        self.selection_strategy = strategy if strategy is not None else statics.NODEPIECE_STRATEGY
        self.num_anchors = num_anchors if num_anchors is not None else statics.DEFAULT_NUM_ANCHORS
        self.max_num_paths = max_num_paths if max_num_paths is not None else statics.DEFAULT_NUM_PATHS
        self.graph_params = self.dataloader.dataset + "_" + self.part + "_" \
                            + str(self.num_anchors) + "_" + str(self.max_num_paths) + "_" + str(dataloader.add_inverse)
        self.save_dir = os.path.join(os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                                                               "data"), "Graph_Data"), self.graph_params)
        self.relations_list = sorted(self.graph.unique_relations.tolist())
        self.path_memory = set()
        if os.path.exists(self.save_dir) and self.dataloader.dataset != "test-dataset" and self.use_cached:
            self.load_state()
        else:
            # self.anchors = self.graph.find_anchors(anchor_strategy=self.selection_strategy,
            #                                        num_anchors=self.num_anchors)
            self.anchors = self.graph.find_highest_outdegree_anchors(self.num_anchors)
            self.anchor_set = set(self.anchors)

            # numerical indices for entities and relations - this is required so that both anchors and relations
            # can be embedded using a single embedding layer
            ancs = self.anchors.copy()
            tokens = self.special_tokens.copy()
            self.vocab = ancs + self.relations_list + tokens
            # when concatenating all edges and relations into one vocab we must introduce offsets to avoid collisions
            # between edge ids and node ids
            token2id_ancs = {t: i for i, t in enumerate(ancs)}
            id_offset = max(token2id_ancs.values()) + 1
            token2id_rel = {(self.edge_offset + t): (id_offset + i) for i, t in enumerate(self.relations_list)}
            token2id_ancs.update(token2id_rel)
            sp_token_offset = max(token2id_ancs.values()) + 1
            token_vocab = {t: sp_token_offset + i for i, t in enumerate(tokens)}
            token2id_ancs.update(token_vocab)
            self.token2id = copy.deepcopy(token2id_ancs)
            self.id2token = {v: k for k, v in self.token2id.items()}
            statics.check_and_make_dir(self.save_dir)
            if os.path.exists(self.save_dir) and self.dataloader.dataset != "test-dataset" and self.use_cached:
                self.save_state()
        self.vocab_size = len(self.token2id)
        self.paths = {}

    def load_state(self):
        """
        Load saved anchors, vocab, edge offset and token2id and id2token vocabs
        for reproducibility. Otherwise new anchors are selected and a new model needs to be trained
        :return: None
        """
        with open(os.path.join(self.save_dir, 'anchors.pickle'), 'rb') as f:
            self.anchors = pickle.load(f)
        with open(os.path.join(self.save_dir, 'token2id.pickle'), 'rb') as f:
            self.token2id = pickle.load(f)
        with open(os.path.join(self.save_dir, 'id2token.pickle'), 'rb') as f:
            self.id2token = pickle.load(f)
        with open(os.path.join(self.save_dir, 'vocab.pickle'), 'rb') as f:
            self.vocab = pickle.load(f)
        with open(os.path.join(self.save_dir, 'edge_offset.pickle'), 'rb') as f:
            self.edge_offset = pickle.load(f)
        if os.path.exists(os.path.join(self.save_dir, 'paths.pickle')):
            with open(os.path.join(self.save_dir, 'paths.pickle'), 'rb') as f:
                self.paths = pickle.load(f)

    def save_state(self):
        """
        Save the anchors, vocab, edge offset and token2id and id2token vocabs
        for reproducibility.
        :return: None
        """
        with open(os.path.join(self.save_dir, 'anchors.pickle'), 'wb') as f:
            pickle.dump(self.anchors, file=f)
        with open(os.path.join(self.save_dir, 'token2id.pickle'), 'wb') as f:
            pickle.dump(self.token2id, file=f)
        with open(os.path.join(self.save_dir, 'id2token.pickle'), 'wb') as f:
            pickle.dump(self.id2token, file=f)
        with open(os.path.join(self.save_dir, 'vocab.pickle'), 'wb') as f:
            pickle.dump(self.vocab, file=f)
        with open(os.path.join(self.save_dir, 'edge_offset.pickle'), 'wb') as f:
            pickle.dump(self.edge_offset, file=f)

    def save_paths(self):
        with open(os.path.join(self.save_dir, 'paths.pickle'), 'wb') as f:
            pickle.dump(self.paths, file=f)

    def get_paths(self, node_ids: Iterable):
        """
        Returns the shortest paths for the node (or nodes) to the closest anchors
        :param node_ids: An iterable of node_ids to find paths to
        :return: A Tuple[bool, Dict] containing True if a path was found for all nodes or False otherwise
        and a Dict[node_id : List[paths[entity ids]]]
        """
        paths = {}
        for node in node_ids:
            valid_paths = []
            if node not in self.paths.keys():
                raw_paths = self.graph.get_paths(target_node=node, anchors=self.anchors,
                                                 mode="path", path_type="shortest",
                                                 path_num_limit=self.max_num_paths, exclude_self=False)
                for node_id in raw_paths.keys():
                    node_paths = raw_paths[node_id]
                    for path in node_paths:
                        # if path is empty check next path
                        if path[0] == statics.NOTHING_TOKEN:
                            continue
                        else:
                            # if path is not empty, change every token with its id and
                            # append to valid paths - this is to enable indexing of the embedding
                            # tables of the model
                            id_path = []
                            for i, token in enumerate(path):
                                if i == 0:
                                    id_path.append(self.token2id[token])
                                else:
                                    id_path.append(self.token2id[token + self.edge_offset])
                            valid_paths.append(id_path)
                    # if there is at least one valid path append it to the path dict
                    # and return the paths to the caller
                    if len(valid_paths) > 0:
                        self.paths[node] = valid_paths
                        paths[node] = self.paths[node]
                        return True, paths
                    else:
                        # return to the caller to sample another set of nodes
                        return False, {}
            else:
                return True, {node: self.paths[node]}

    def get_unique_path(self,
                        node_ids):  # TODO get the path dict which now containes edge objects instead of edge types and get edge_type to get edge id or .tuple to get the edge and remove it/add it
        """
        Returns the shortest paths for the node (or nodes) to the closest anchors
        :param node_ids: An iterable of node_ids to find paths to
        :return: A Tuple[bool, Dict] containing True if a path was found for all nodes or False otherwise
        and a Dict[node_id : List[paths[entity ids]]]
        """
        paths = {}
        for node in node_ids:
            valid_paths = []
            if node not in self.paths.keys():  ### FIND NEW PATHS
                raw_paths = self.graph.get_paths_with_indices(target_node=node, anchors=self.anchors,
                                                              mode="path", path_type="shortest",
                                                              path_num_limit=self.max_num_paths, exclude_self=False)
                for node_id in raw_paths.keys():  # for each node
                    edges_to_add = []
                    node_paths = raw_paths[node_id]
                    selected_path = []
                    shortest_path = []
                    found_shortest = False
                    for idx, path in enumerate(node_paths):  # for each path of the node
                        # if path is empty check next path
                        if path[0] == statics.NOTHING_TOKEN:
                            continue
                        else:
                            if not found_shortest: # cache the shortest path in case a unique cannot be found
                                shortest_path = [path[0]] + [
                                    self.graph.graph.es[i]['edge_type'] for i in path[1:]]
                            # create edge type path
                            edge_type_path = [path[0]] + [self.graph.graph.es[i]['edge_type'] for i in path[1:]]
                            # check if path has already been used
                            path_string = str(edge_type_path)
                            if path_string in self.path_memory and idx < (len(node_paths) - 1):
                                # if path has been used and there are more paths, check next path
                                continue
                            elif path_string in self.path_memory and idx == (len(node_paths) - 1):
                                # if path has been used and there are no more paths
                                # improvise
                                removed_edges = set()
                                found_path = False
                                # get the tuples and types of the edges in all paths
                                # so they can be located after removing edges
                                # from the graph, which causes reindexing
                                tuple_paths = []
                                type_paths = []
                                for index_path in node_paths:
                                    type_path = []
                                    tuple_path = []
                                    for index in index_path[1:]:
                                        tuple_path.append(self.graph.graph.es[index].tuple)
                                        type_path.append(self.graph.graph.es[index]['edge_type'])
                                    tuple_paths.append(tuple_path)
                                    type_paths.append(type_path)

                                # for each path in node_paths
                                for j, path_to_corrupt in enumerate(node_paths):
                                    if found_path: # if a unique path ahs been found stop the iterations
                                        break
                                    for l, edge in enumerate(path_to_corrupt[1:]):
                                        # get the frequency of each edge type and select the highest frequency one
                                        most_frequent_edge_index = -100
                                        max_freq = -10
                                        for k, e in enumerate(path_to_corrupt[1:]): # the first element is the anchor
                                            if e in removed_edges:# if the edge has already been removed skip it
                                                continue
                                            edge_freq = self.dataloader.edge_stats['train']['edge_counts'][
                                                type_paths[j][l]]
                                            if edge_freq > max_freq:
                                                most_frequent_edge_index = k  # k + 1
                                        if most_frequent_edge_index < 0:
                                            continue
                                        edge_to_remove = path_to_corrupt[most_frequent_edge_index + 1] # the index
                                        edge_to_remove_tuple = tuple_paths[j][most_frequent_edge_index]
                                        edge_to_remove_type = type_paths[j][most_frequent_edge_index]
                                        removed_edges.add(edge_to_remove)
                                        # find the edge based on its start, end and type
                                        removal_candidate = self.graph.graph.es.find(_source_eq=edge_to_remove_tuple[0],
                                                                               _target_eq=edge_to_remove_tuple[1],
                                                                               edge_type_eq=edge_to_remove_type)
                                        # get the properties of the edge that is removed so we can add it back later
                                        edge_properties = removal_candidate.attributes()
                                        edge_tuple = removal_candidate.tuple
                                        edges_to_add.append({'tuple': edge_tuple, 'properties': edge_properties})
                                        edge_id = removal_candidate.index
                                        # remove it from the graph
                                        self.graph.graph.delete_edges([edge_id])
                                        # find all paths to all anchors again
                                        raw_extended_paths = self.graph.get_paths_with_indices(target_node=node_id,
                                                                                               anchors=self.anchors,
                                                                                               mode="path",
                                                                                               path_type="shortest",
                                                                                               path_num_limit=self.max_num_paths,
                                                                                               exclude_self=False)
                                        # iterate over them to check uniqueness
                                        new_paths = raw_extended_paths[node_id]
                                        for ind, new_path in enumerate(new_paths):
                                            # if path is empty check next path
                                            if new_path[0] == statics.NOTHING_TOKEN:
                                                continue
                                            else:
                                                # create edge type path
                                                new_edge_type_path = [new_path[0]] + [self.graph.graph.es[i]['edge_type'] for
                                                                                      i in
                                                                                      new_path[1:]]
                                                # check if path has already been used
                                                path_string = str(new_edge_type_path)
                                                if path_string in self.path_memory:
                                                    # if path has been used and there are more paths, check next path
                                                    continue
                                                else:
                                                    self.path_memory.add(path_string)
                                                    found_path = True
                                                    selected_path = new_edge_type_path
                                                    break
                                        if found_path:
                                            break
                            else:  # if not, memoize it
                                self.path_memory.add(path_string)
                                selected_path = edge_type_path
                            # if no unique path was found and there is at least one path, add it
                            if len(selected_path) == 0 and len(shortest_path) > 0:
                                selected_path = shortest_path

                            # change every token with its id and
                            # append to valid paths - this is to enable indexing of the embedding
                            # tables of the model
                            id_path = []
                            if len(selected_path) > 0:
                                for i, token in enumerate(selected_path):
                                    if i == 0:
                                        id_path.append(self.token2id[token])
                                    else:
                                        id_path.append(self.token2id[token + self.edge_offset])
                                valid_paths.append(id_path)
                                break
                    # ad the edges back to the graph
                    for removed_edge in edges_to_add:
                        self.graph.graph.add_edges([removed_edge['tuple']], attributes=removed_edge['properties'])
                    # if there is at least one valid path append it to the path dict
                    # and return the paths to the caller
                    if len(valid_paths) > 0:
                        self.paths[node] = valid_paths
                        paths[node] = self.paths[node]
                        return True, paths
                    else:
                        # return to the caller to sample another set of nodes
                        return False, {}
            else:
                return True, {node: self.paths[node]}

    def memoize_path(self, path):
        self.path_memory.add(str(path))

    # def get_unique_path(self, node_ids):
    #     # get all shortest paths sorted by cost/hops/weight
    #     # start from lowest cost and check if path is memoized
    #     # if not memoize and return
    #     # if passed through all and no path found
    #     # get first path and
    #     pass

    def token_id(self, token, token_type):
        """
        The zero-indexed token id for the anchor or relation
        :param token_type: The type of token. One of ['anchor', 'edge','special']
        :param token: an anchor or relation id
        :return: the zero-indexed id
        """
        if token_type == 'anchor':
            return self.token2id[token]
        elif token_type == 'edge':
            return self.token2id[token + self.edge_offset]
        else:
            return self.token2id[token]

    def id_token(self, token_id, token_type):
        """
        The id of the element
        :param token_type: The type of token. One of ['anchor', 'edge','special']
        :param token_id: the zero-indexed id
        :return: the actual id in the graph
        """
        if token_type == 'edge':
            return self.id2token[token_id] - self.edge_offset
        else:
            return self.id2token[token_id]

    def save_num_unique_paths(self, epoch: int):
        """
        Calculate and save the number of unique paths in the path dictionary of the class
        :param epoch: The epoch of training that this count corresponds to (int)
        :return:
        """
        save_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "Data"), "Graph_Data"), "Path_Stats")
        statics.check_and_make_dir(save_dir)
        save_path = os.path.join(save_dir, "path_counts_" + str(self.num_anchors) + "_" + str(self.dataloader.dataset) +
                                 "_" + str(self.max_num_paths) + ".csv")
        if not os.path.exists(save_path):
            with open(save_path, 'w') as f:
                f.write("Epoch;Unique_paths\n")
        # create a list of all the paths contained in the paths dictionary as strings
        total_paths = [str(path) for paths in self.paths.values() for path in paths]
        with open(save_path, 'a+') as f:
            unique = set(total_paths)  # convert the list to a set
            f.write(str(epoch) + ";" + str(len(unique)) + "\n")

    @property
    def get_graph(self):
        return self.graph

    @property
    def get_anchors(self):
        return self.anchors

    @property
    def get_vocab(self):
        return self.vocab

    @property
    def vocab_len(self):
        return self.vocab_size
