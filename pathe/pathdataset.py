import warnings
import random
from sys import getsizeof
from handler import Handler
from kgloader import KgLoader
import torch
import json
import os


# TODO 4 or 5 paths for each
# Further corrective measures
class PathDataset:

    def __init__(self, dataset: KgLoader, num_paths_per_entity: int, num_steps:
    int,
        parallel=False):
        """
        The constructor
        @param dataset:  The object that contains the dataset triples
        generation
        @type dataset: KgLoader
        @param parallel:  whether to use parallelization
        @type parallel: bool
        """
        self.dataset = dataset
        self.parallel = parallel
        self.num_paths_per_entity = num_paths_per_entity
        self.num_steps = num_steps
        self.graph = {'train': Handler(dataloader=dataset, part='train'),
                      'val': Handler(dataloader=dataset,
                                     part='validation'),
                      'test': Handler(dataloader=dataset, part='test')
                      }
        self.verbalizer = Verbalizer(self.dataset)
        self.dataset_name = self.dataset.dataset
        self.dataset_dir = os.path.join(os.path.join(
            os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                         "data"), "path_datasets"), self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        for part in self.graph:
            if not os.path.exists(os.path.join(self.dataset_dir, part)):
                os.makedirs(os.path.join(self.dataset_dir, part))
        # self.print_degree_distributions()
        self.paths = {}
        self.between_paths = {}
        self.current_paths = None
        self.num_paths = None
        self.relational_context = None
        self.ht_relation = {}
        self.make_relational_context_dataset()
        # self.make_path_between_dataset(num_paths=10)
        self.make_random_walk_dataset(num_paths=self.num_paths_per_entity,
                                      num_steps=self.num_steps,
                                      parallel=self.parallel)
        self.create_target_csv()
        # self.verbalize_paths(5)  # fix add check that there are as many paths


    def make_path_between_dataset(self,num_paths=10):
        for part in self.graph:
            graph = self.graph[part]
            if part == 'train':
                triples = self.dataset.train_triples
            elif part == 'val':
                triples = self.dataset.val_triples
            else:
                triples = self.dataset.test_triples
            dirname = os.path.join(self.dataset_dir, part)
            self.between_paths[part] = (
                graph.graph.get_shortest_k_paths_between(triples,
                                                            num_paths=10))
            self.save_betweeness_dataset_in_path_csv(num_paths=num_paths,
                                                   part=part)

    def save_betweeness_dataset_in_path_csv(self, part, num_paths):
        """
        A function that saves the randm walk paths in csv format with space as
        separator
        -------
        The function produces four csv files:
        1. a pathER csv which contains the paths in start_node, end_node,
        path format
        2. a pathRe csv which contains only the relations in the path
        3. a pathEn csv which contains only the entities in the path
        4. a pathJoined csv which contains the joined entity-relation path
        -------
        @param part: The part of the dataset (rain,validation or test)
        @type part: String
        @param num_paths: The number of paths produced per node
        @type num_paths: int
        @return: Nothing
        @rtype: None
        """
        if not self.between_paths:
            raise Exception("The are no paths to save.")
        part = part
        with open(os.path.join(os.path.join(self.dataset_dir, part),
                               str(num_paths) + "_" +
                               'bt_paths.csv'),
                  'w+') as k:
            k.write("relation_path,entity_path\n")
            for entity in self.between_paths[part].keys():
                paths = self.between_paths[part][entity]
                paths_Re = paths[0]
                paths_En = paths[1]
                for i in range(len(paths_Re)):
                    relation_path = paths_Re[i]
                    entity_path = paths_En[i]
                    relation_path_str = ""
                    entity_path_str = ""
                    for step in relation_path:
                        relation_path_str = relation_path_str + str(
                            step) + " "
                    for node in entity_path:
                        entity_path_str = entity_path_str + str(node) + " "
                    relation_path_str = relation_path_str.strip() + "\n"
                    entity_path_str = entity_path_str.strip() + "\n"
                    k.write(relation_path_str.strip("\n") + "," +
                            entity_path_str)



    def print_degree_distributions(self):
        with open(os.path.join(self.dataset_dir, 'stats.csv'), 'w+') as f:
            for part in self.graph:
                f.write(part + ";in;")
                d = self.graph[part].graph.degree_distribution(mode='in').bins()
                for i in d:
                    f.write(str(i[2]) + ";")
                f.write("\n")
                f.write(part + ";out;")
                d = self.graph[part].graph.degree_distribution(
                    mode='out').bins()
                for i in d:
                    f.write(str(i[2]) + ";")
                f.write("\n")
        print("done")

    def make_shortest_path_dataset(self, k: int = 1):
        """
        A function that calls the graph handler to find all shortest paths and
        assigns the shortest paths to the paths variable
        @param k:  The number of shortest paths to generate (default 1)
        @type k: int
        @return: None
        @rtype: None
        """
        if k < 1:
            raise Exception("k must be at least 1.")
        if k > 1:
            for part in self.graph:
                graph = self.graph[part]
                self.paths[part] = graph.graph.get_shortest_k_paths(num_paths=k)
            self.num_paths = k
        else:
            if self.dataset.dataset == 'codex-small':
                for part in self.graph:
                    graph = self.graph[part]
                    self.paths[part] = graph.graph.get_shortest_paths_to_nodes()
            else:
                for part in self.graph:
                    graph = self.graph[part]
                    self.paths[
                        part] = graph.graph.get_shortest_paths_to_nodes() # nodes=graph.anchors
            self.num_paths = 1

    def make_random_walk_dataset(self, num_paths: int = 50,
                                 num_steps: int = 20, parallel: bool = False):
        """
        A function that generates random walk over the graph for training
        @param num_paths: The number of paths per node to generate
        @type num_paths: int
        @param num_steps: The number of steps that each path should take
        @type num_steps: int
        @return: None
        @rtype: None
        """
        for part in self.graph:
            graph = self.graph[part]
            if parallel:
                self.paths[part] = graph.graph.get_parallel_random_walks(
                    num_paths=num_paths, num_steps=num_steps, part=part,
                                                            mixing_ratio=0.5,
                                                            max_attempts=1000)

            else:
                self.paths[part] = graph.graph.get_random_walks(num_paths=num_paths,
                                                                num_steps=num_steps,
                                                                part=part,
                                                                mixing_ratio=0.5,
                                                                max_attempts=1000)
            self.save_rnd_walk_dataset_in_path_csv(num_paths=num_paths,
                                                   num_steps=num_steps,
                                                   part=part)
            self.save_triple_tensor_and_dicts(part)

    def make_simple_path_dataset(self, batch_size_start: int = 1,
                                 batch_size_end: int = 1):
        for part in self.graph:
            graph = self.graph[part]
            unique_nodes = graph.dataloader.unique_entities.tolist()
            for i in range(0, len(unique_nodes), batch_size_start):
                start_nodes = unique_nodes[i:i + batch_size_start]
                for j in range(0, len(unique_nodes), batch_size_end):
                    end_nodes = unique_nodes[j:j + batch_size_end]
                    if not self.current_paths:
                        self.current_paths = \
                            graph.graph.get_simple_paths_between_nodes(
                                start_nodes=start_nodes, target_nodes=end_nodes)
                    else:
                        self.current_paths.update(
                            graph.graph.get_simple_paths_between_nodes(
                                start_nodes=start_nodes,
                                target_nodes=end_nodes))
                    if self.current_paths and \
                            getsizeof(self.current_paths) > 1e9:
                        self.save_dataset_in_chunked_path_csv(i, j, part)
                        self.current_paths = None

    def verbalize_paths(self, num_paths: int = 10):
        """
        A function that verbalizes a number of paths used for training
        @param num_paths: The number of paths to verbalize. The function will
        verbalize num_paths for num_paths nodes so the total paths verbalized
        will be num_paths*num_paths
        @type num_paths: int
        @return: None
        @rtype: None
        """
        remaining = num_paths
        for entity in self.paths['train'].keys():
            if remaining == 0:
                break
            paths = self.paths['train'][entity]
            paths_Re = paths[0]
            paths_En = paths[1]
            paths_Joined = paths[2]
            for i in range(num_paths):
                if i < len(paths_Re):
                    rel_path = paths_Re[i]
                    ent_path = paths_En[i]
                    joint_path = paths_Joined[i]
                    rel_verbal = self.verbalizer.verbalize_path(rel_path,
                                                                'relational')
                    ent_verbal = self.verbalizer.verbalize_path(ent_path,
                                                                'entity')
                    joint_verbal = self.verbalizer.verbalize_path(joint_path,
                                                                  'joined')
                    print('relational:' + rel_verbal)
                    print('entity:' + ent_verbal)
                    print('joined:' + joint_verbal)
                else:
                    break
            remaining -= 1

    def make_relational_context_dataset(self):
        """
        A function that makes the relational context dataset
        @return:
        @rtype:
        """
        for part in self.graph:
            # if not os.path.exists(
            #         os.path.join(os.path.join(self.dataset_dir,
            #                                   part), 'relational_context.csv')):
                self.relational_context = self.graph[
                    part].graph.get_relational_context()
                if part == 'train':
                    self.save_train_relational_context()
                else:
                    self.save_relational_context(part)

    def save_relational_context(self, part):
        """
        A function that saves the relational context dataset in csv format
        of the form "node_id,direction,[edges separated by space]" for the
        test and val sets.
        @return: None
        @rtype: None
        """
        if not self.relational_context:
            raise Exception("No relational context data found.")
        with open(os.path.join(os.path.join(self.dataset_dir, part),
                               'relational_context_' + part + '.csv'),
                  'w+') as f:
            f.write("node_id,direction,edges\n")
            for node_id in self.relational_context.keys():
                node_context_dict = self.relational_context[node_id]
                incoming_edges = node_context_dict['in']
                outgoing_edges = node_context_dict['out']
                incoming_line = str(node_id) + ",in,"
                outgoing_line = str(node_id) + ",out,"
                if len(incoming_edges) > 0:
                    for edge in incoming_edges:
                        incoming_line += str(edge) + " "
                    incoming_line.strip()
                    f.write(incoming_line + "\n")
                if len(outgoing_edges) > 0:
                    for edge in outgoing_edges:
                        outgoing_line += str(edge) + " "
                    outgoing_line.strip()
                    f.write(outgoing_line + "\n")

    def save_train_relational_context(self):
        """
        A function that saves the relational context dataset of the train set
        in csv format of the form "node_id,direction,[edges separated by space]"
        This is saved in all the train,test and val directories and used for
        test and val as well.
        @return: None
        @rtype: None
        """
        if not self.relational_context:
            raise Exception("No relational context data found.")
        with open(os.path.join(os.path.join(self.dataset_dir, 'train'),
                               'relational_context.csv'),
                  'w+') as f, \
                open(os.path.join(os.path.join(self.dataset_dir, 'test'),
                                  'relational_context.csv'),
                     'w+') as r, \
                open(os.path.join(os.path.join(self.dataset_dir, 'val'),
                                  'relational_context.csv'),
                     'w+') as e:
            f.write("node_id,direction,edges\n")
            r.write("node_id,direction,edges\n")
            e.write("node_id,direction,edges\n")
            for node_id in self.relational_context.keys():
                node_context_dict = self.relational_context[node_id]
                incoming_edges = node_context_dict['in']
                outgoing_edges = node_context_dict['out']
                incoming_line = str(node_id) + ",in,"
                outgoing_line = str(node_id) + ",out,"
                if len(incoming_edges) > 0:
                    for edge in incoming_edges:
                        incoming_line += str(edge) + " "
                    incoming_line.strip()
                    f.write(incoming_line + "\n")
                    r.write(incoming_line + "\n")
                    e.write(incoming_line + "\n")
                if len(outgoing_edges) > 0:
                    for edge in outgoing_edges:
                        outgoing_line += str(edge) + " "
                    outgoing_line.strip()
                    f.write(outgoing_line + "\n")
                    r.write(outgoing_line + "\n")
                    e.write(outgoing_line + "\n")

    def save_dataset_in_chunked_path_csv(self, current_index, chunk_number,
                                         part):
        """
        A function that saves the paths in csv format with space as separator
        -------
        The function produces four csv files:
        1. a pathER csv which contains the paths in start_node, end_node,
        path format
        2. a pathRe csv which contains only the relations in the path
        3. a pathEn csv which contains only the entities in the path
        4. a pathJoined csv which contains the joined entity-relation path
        -------
        @return: None
        @rtype: None
        """
        if not self.current_paths:
            raise Exception("The are no paths to save.")
        with open(os.path.join(self.dataset_dir, part + "_" +
                                                 'paths_pathER_' + str(
            current_index) + "_" + str(chunk_number) +
                                                 '.csv'),
                  'w+') as f, \
                open(os.path.join(self.dataset_dir, part + "_" +
                                                    str(self.num_paths) +
                                                    "_" + 'paths_pathRe_'
                                                    + str(
                    current_index) + "_" + str(chunk_number) +
                                                    '.csv'),
                     'w+') as r, \
                open(os.path.join(self.dataset_dir, part + "_" +
                                                    str(self.num_paths) +
                                                    "_" + 'paths_pathEn_'
                                                    + str(
                    current_index) + "_" + str(chunk_number) +
                                                    '.csv'),
                     'w+') as e, \
                open(os.path.join(self.dataset_dir, part + "_" +
                                                    str(self.num_paths) + "_" +
                                                    'paths_pathJoined_' + str(
                    current_index) + "_" + str(chunk_number) +
                                                    '.csv'),
                     'w+') as j:
            f.write("start_node,end_node,path\n")
            for entity in self.paths[part].keys():
                paths = self.paths[part][entity]
                paths_Re = paths[0]
                paths_En = paths[1]
                paths_Joined = paths[2]
                for i in range(len(paths_Re)):
                    relation_path = paths_Re[i]
                    entity_path = paths_En[i]
                    joined_path = paths_Joined[i]
                    start_node = entity_path[0]
                    end_node = entity_path[-1]
                    path_ER = str(start_node) + "," + str(end_node) + ","
                    relation_path_str = ""
                    entity_path_str = ""
                    joined_path_str = ""
                    for step in relation_path:
                        path_ER = path_ER + str(step) + " "
                        relation_path_str = relation_path_str + str(step) + " "
                    for node in entity_path:
                        entity_path_str = entity_path_str + str(node) + " "
                    for element in joined_path:
                        joined_path_str = joined_path_str + str(element) + " "

                    path_ER = path_ER.strip() + "\n"
                    relation_path_str = relation_path_str.strip() + "\n"
                    entity_path_str = entity_path_str.strip() + "\n"
                    joined_path_str = joined_path_str.strip() + "\n"

                    f.write(path_ER)
                    r.write(relation_path_str)
                    e.write(entity_path_str)
                    j.write(joined_path_str)

    def save_dataset_in_path_csv(self, part):
        """
        A function that saves the paths in csv format with space as separator
        -------
        The function produces four csv files:
        1. a pathER csv which contains the paths in start_node, end_node,path format
        2. a pathRe csv which contains only the relations in the path
        3. a pathEn csv which contains only the entities in the path
        4. a pathJoined csv which contains the joined entity-relation path
        -------
        @return: None
        @rtype: None
        """
        if not self.paths:
            raise Exception("The are no paths to save.")
        with open(os.path.join(self.dataset_dir, part + "_" +
                                                 str(self.num_paths) +
                                                 "_" + 'paths_pathER.csv'),
                  'w+') as f, \
                open(os.path.join(self.dataset_dir, part + "_" +
                                                    str(self.num_paths) + "_" +
                                                    'paths_pathRe.csv'),
                     'w+') as r, \
                open(os.path.join(self.dataset_dir, part + "_" +
                                                    str(self.num_paths) + "_" +
                                                    'paths_pathEn.csv'),
                     'w+') as e, \
                open(os.path.join(self.dataset_dir, part + "_" +
                                                    str(self.num_paths) + "_" +
                                                    'paths_pathJoined.csv'),
                     'w+') as j:
            f.write("start_node,end_node,path\n")
            for entity in self.paths[part].keys():
                paths = self.paths[part][entity]
                paths_Re = paths[0]
                paths_En = paths[1]
                paths_Joined = paths[2]
                for i in range(len(paths_Re)):
                    relation_path = paths_Re[i]
                    entity_path = paths_En[i]
                    joined_path = paths_Joined[i]
                    start_node = entity_path[0]
                    end_node = entity_path[-1]
                    path_ER = str(start_node) + "," + str(end_node) + ","
                    relation_path_str = ""
                    entity_path_str = ""
                    joined_path_str = ""
                    for step in relation_path:
                        path_ER = path_ER + str(step) + " "
                        relation_path_str = relation_path_str + str(step) + " "
                    for node in entity_path:
                        entity_path_str = entity_path_str + str(node) + " "
                    for element in joined_path:
                        joined_path_str = joined_path_str + str(element) + " "

                    path_ER = path_ER.strip() + "\n"
                    relation_path_str = relation_path_str.strip() + "\n"
                    entity_path_str = entity_path_str.strip() + "\n"
                    joined_path_str = joined_path_str.strip() + "\n"

                    f.write(path_ER)
                    r.write(relation_path_str)
                    e.write(entity_path_str)
                    j.write(joined_path_str)

    def save_rnd_walk_dataset_in_path_csv(self, num_paths, num_steps, part):
        """
        A function that saves the randm walk paths in csv format with space as
        separator
        -------
        The function produces four csv files:
        1. a pathER csv which contains the paths in start_node, end_node,
        path format
        2. a pathRe csv which contains only the relations in the path
        3. a pathEn csv which contains only the entities in the path
        4. a pathJoined csv which contains the joined entity-relation path
        -------
        @param part: The part of the dataset (rain,validation or test)
        @type part: String
        @param num_paths: The number of paths produced per node
        @type num_paths: int
        @param num_steps: The number of steps per path
        @type num_paths: int
        @return: Nothing
        @rtype: None
        """
        if not self.paths:
            raise Exception("The are no paths to save.")
        part = part
        with open(os.path.join(os.path.join(self.dataset_dir, part),
                               str(num_paths) + "_" + str(
                                   num_steps) + "_" +
                               'er_paths.csv'),
                  'w+') as k:
            k.write("relation_path,entity_path\n")
            for entity in self.paths[part].keys():
                paths = self.paths[part][entity]
                paths_Re = paths[0]
                paths_En = paths[1]
                for i in range(len(paths_Re)):
                    relation_path = paths_Re[i]
                    entity_path = paths_En[i]
                    relation_path_str = ""
                    entity_path_str = ""
                    for step in relation_path:
                        relation_path_str = relation_path_str + str(step) + " "
                    for node in entity_path:
                        entity_path_str = entity_path_str + str(node) + " "
                    relation_path_str = relation_path_str.strip() + "\n"
                    entity_path_str = entity_path_str.strip() + "\n"
                    k.write(relation_path_str.strip("\n") + "," +
                            entity_path_str)

    def save_triple_tensor_and_dicts(self, part):
        """
        A function that saves the triples od each part of the dataset as
        torch.tensors and also saves the id2entity and id2relation dicts as
        JSON for each of the splits
        Parameters
        ----------
        part : The part of the dataset in ['train', 'test', 'val']

        Returns
        -------

        """
        if part == 'train':
            triples = self.dataset.train_triples
            dirname = os.path.join(self.dataset_dir, part)
            torch.save(triples, os.path.join(dirname, 'triples.pt'))
            id2entity = self.dataset.triple_factory.factory_dict[
                'training'].entity_id_to_label
            id2relation = dataset.triple_factory.factory_dict[
                'training'].relation_id_to_label
            with open(os.path.join(dirname, 'id2entity.json'), 'w') as f:
                json.dump(id2entity, f)
            with open(os.path.join(dirname, 'id2relation.json'), 'w') as f:
                json.dump(id2relation, f)

        elif part == 'test':
            triples = self.dataset.test_triples
            dirname = os.path.join(self.dataset_dir, part)
            torch.save(triples, os.path.join(dirname, 'triples.pt'))
            id2entity = self.dataset.triple_factory.factory_dict[
                'testing'].entity_id_to_label
            id2relation = self.dataset.triple_factory.factory_dict[
                'testing'].relation_id_to_label
            with open(os.path.join(dirname, 'id2entity.json'), 'w') as f:
                json.dump(id2entity, f)
            with open(os.path.join(dirname, 'id2relation.json'), 'w') as f:
                json.dump(id2relation, f)
        else:
            triples = self.dataset.val_triples
            dirname = os.path.join(self.dataset_dir, part)
            torch.save(triples, os.path.join(dirname, 'triples.pt'))
            id2entity = self.dataset.triple_factory.factory_dict[
                'validation'].entity_id_to_label
            id2relation = dataset.triple_factory.factory_dict[
                'validation'].relation_id_to_label
            with open(os.path.join(dirname, 'id2entity.json'), 'w') as f:
                json.dump(id2entity, f)
            with open(os.path.join(dirname, 'id2relation.json'), 'w') as f:
                json.dump(id2relation, f)

    def create_target_csv(self):
        """
        Created the {'h,t' : k-hot encoded target torch vector} dict
        Returns
        -------

        """
        for part in ['train', 'val', 'test']:
            if part == 'train':
                for i in range(self.dataset.train.shape[0]):
                    idx = str(self.dataset.train[i, 0].item()) + "," + str(
                        self.dataset.train[i, 2].item()) + ","
                    if idx not in self.ht_relation:
                        self.ht_relation[idx] = [
                            self.dataset.train[i, 1].item()]
                    else:
                        self.ht_relation[idx].append(
                            self.dataset.train[i, 1].item())
                # for key in self.ht_relation:
                #     target = torch.zeros(num_relations)
                #     target[self.ht_relation[key]] = 1
                #     self.ht_targets[key] = target

                with (open(os.path.join(os.path.join(self.dataset_dir, "train"),
                                        "rel_neighbors.csv"), "w+")) as f:
                    f.write("Head,Tail,Relations\n")
                    for key in self.ht_relation:
                        row = key
                        for relation in self.ht_relation[key]:
                            row += str(relation) + " "
                        row = row.strip()
                        f.write(row + "\n")
            elif part == 'val':
                for i in range(self.dataset.validation.shape[0]):
                    idx = str(self.dataset.validation[i, 0].item()) + "," + str(
                        self.dataset.validation[i, 2].item()) + ","
                    if idx not in self.ht_relation:
                        self.ht_relation[idx] = [
                            self.dataset.validation[i, 1].item()]
                    else:
                        self.ht_relation[idx].append(
                            self.dataset.validation[i, 1].item())
                # for key in self.ht_relation:
                #     target = torch.zeros(num_relations)
                #     target[self.ht_relation[key]] = 1
                #     self.ht_targets[key] = target

                with (open(os.path.join(os.path.join(self.dataset_dir, "val"),
                                        "rel_neighbors.csv"), "w+")) as f:
                    f.write("Head,Tail,Relations\n")
                    for key in self.ht_relation:
                        row = key
                        for relation in self.ht_relation[key]:
                            row += str(relation) + " "
                        row = row.strip()
                        f.write(row + "\n")
            else:
                for i in range(self.dataset.test.shape[0]):
                    idx = str(self.dataset.test[i, 0].item()) + "," + str(
                        self.dataset.test[i, 2].item()) + ","
                    if idx not in self.ht_relation:
                        self.ht_relation[idx] = [
                            self.dataset.test[i, 1].item()]
                    else:
                        self.ht_relation[idx].append(
                            self.dataset.test[i, 1].item())
                # for key in self.ht_relation:
                #     target = torch.zeros(num_relations)
                #     target[self.ht_relation[key]] = 1
                #     self.ht_targets[key] = target

                with (open(os.path.join(os.path.join(self.dataset_dir, "test"),
                                        "rel_neighbors.csv"), "w+")) as f:
                    f.write("Head,Tail,Relations\n")
                    for key in self.ht_relation:
                        row = key
                        for relation in self.ht_relation[key]:
                            row += str(relation) + " "
                        row = row.strip()
                        f.write(row + "\n") \
 \
                            # TOFIX: The components of the shortest paths are shortest paths themselves



# so there is nothing to replace, each pair of nodes had only one path
# between them in the entire dataset. POSSIBLE FIX: Group paths based on
# start node and end node and when replacing a step, finding a random
# shortest path from the start node to some other node and the path from the
# end node to the end node of the original path (if it exists)
class PathXT(PathDataset):
    """
    A class that extends paths
    """

    def __init__(self, dataset: KgLoader):
        """
        The constructor
        @param dataset:  The object that contains the dataset triples
        generation
        @type dataset: DataLoader
        """
        super().__init__(dataset)
        self.er_components = {}
        self.r_components = {}
        self.make_shortest_path_dataset(k=1)
        self.build_composition_dicts()
        self.current_node = 0
        # self.save_dataset_in_path_csv()

    def get_paths(self, num_paths: int, xtension_prob: float = 0.2):
        """
        A function that returns extended paths
        @param num_paths: The number of paths to produce
        @type num_paths: int
        @param xtension_prob: The probability with which to replace a step with
        an extension
        @type xtension_prob: float
        @return:
        @rtype:
        """
        if num_paths <= 0:
            warnings.warn("The number of paths was " + str(num_paths) +
                          " and was set to 1.")
            num_paths = 1
        for i in range(num_paths):
            if self.current_node == len(self.paths):
                self.current_node = 0
            if self.current_node not in self.paths:
                self.current_node += 1
                continue
            paths_er = self.paths[self.current_node][2]
            path = list(random.choice(paths_er))
            print("Original path: " + str(path))
            entities = path[0::2]
            it = iter(entities)
            pairs = list(zip(it, it))
            for j in range(len(pairs)):
                if random.random() <= xtension_prob:
                    choices = self.er_components.get(str(pairs[j]))
                    if choices:
                        extension = list(random.choice(choices))
                        path = path[:2 * j] + extension + path[2 * j + 3:]
                    else:
                        continue
            self.current_node += 1
            print("New path: " + str(path))

    def build_composition_dicts(self):
        """
        A function that builds dicts with paths that can replace a single step
        @return: None
        @rtype: None
        """
        for entity in self.paths.keys():
            paths = self.paths[entity]
            paths_er = paths[2]
            for i in range(len(paths_er)):
                path = paths_er[i]
                for j in range(0, len(path), 2):
                    rem = path[j:]
                    for k in range((len(rem) // 2) - 1):
                        component = rem[0:(4 + k * 2) + 1]
                        key = (component[0], component[-1])
                        if key not in self.er_components:
                            self.er_components[str(key)] = set()
                            self.er_components[str(key)].add(tuple(component))
                        else:
                            self.er_components[str(key)].add(tuple(component))
                            print("happened")
                        if key not in self.r_components:
                            self.r_components[str(key)] = set()
                            self.r_components[str(key)].add(tuple(component[
                                                                  1::2]))
                        else:
                            self.r_components[str(key)].add(tuple(component[
                                                                  1::2]))
                            print("happened")


class Verbalizer:
    """
    A class that converts paths of ids to paths of labels
    """

    def __init__(self, dataset: KgLoader):
        """
        The constructor
        @param dataset:The dataset as a DataLoader object
        @type dataset: DataLoader
        """
        self.dataset = dataset
        self.id2entity = dataset.triple_factory.factory_dict[
            'training'].entity_id_to_label
        self.id2relation = dataset.triple_factory.factory_dict[
            'training'].relation_id_to_label

    def verbalize_entity(self, entity_id):
        """
        return the label of a single entity
        @param entity_id: the id of the entity
        @type entity_id: int
        @return: the label
        @rtype: str
        """
        try:
            return self.id2entity[entity_id]
        except KeyError:
            Exception("No such entity id in the dictionary.")

    def verbalize_relation(self, relation_id):
        """
        returns the label of a single relation
        @param relation_id: the id of the relation
        @type relation_id: int
        @return: the label of the relation
        @rtype: str
        """
        try:
            return self.id2relation[relation_id]
        except KeyError:
            Exception("No such relation id in the dictionary.")

    def verbalize_path(self, path, path_type: str = None):
        """
        A function that converts an id path into a lebel path, with each
        label separated by a comma
        @param path: the id path
        @type path: List
        @param path_type: The type of path. Can be "relational", "entity",
        "joined"
        @type path_type: string
        @return: The verbalization
        @rtype: string
        """
        if path_type == 'relational':
            verbalization = ""
            for step in path:
                verbalization += str(self.id2relation[step]) + ", "
        elif path_type == 'entity':
            verbalization = ""
            for step in path:
                verbalization += str(self.id2entity[step]) + ", "
        else:
            verbalization = ""
            relations = path[1::2]
            entities = path[0::2]
            verbalization += str(self.id2entity[entities[0]]) + ", "
            for i in range(len(relations)):
                verbalization += str(self.id2relation[relations[i]]) + ", " \
                                                                       "" + \
                                 str(self.id2entity[entities[i + 1]]) + ", "
        verbalization = verbalization.strip(', ')
        return verbalization


# dataset_name = 'fb15k237'
# dataset_name = 'codex-small'
# dataset_name = 'codex-medium'
# dataset_name = 'codex-large'
# dataset_name = 'wn18rr'
dataset_name = 'yago'
# dataset_name = 'wiki5m'
# dataset_name = "test-dataset"
# # dataset_name = 'ogb-wikikg2'
dataset = KgLoader(dataset_name, add_inverse=False)
path_handler = PathDataset(dataset=dataset, num_paths_per_entity=20,
                           num_steps=10, parallel=True)
print(f'datasets created successfuly in directory {path_handler.dataset_dir}.')
