import os
import torch


def get_dataset_directory(dataset_name):
    dataset_dir = os.path.join(os.path.join(
        os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                     "data"), "path_datasets"), dataset_name)
    return dataset_dir


def load_paths(dataset_dir):
    path_data = {}
    for subset in ["train", "val", "test"]:
        subset_dir = os.path.join(dataset_dir, subset)
        path_file = None
        erpath_pattern = "_paths.csv"
        for subset_file in os.listdir(subset_dir):
            if subset_file.endswith(erpath_pattern):
                path_file = subset_file
        with open(os.path.join(subset_dir, path_file), 'r') as f:
            paths = f.readlines()
            paths = [x.strip('\n').split(',')[0] for x in paths]
            paths.pop(0)
            path_data[subset] = paths
    return path_data


def load_rel_context(dataset_dir):
    combined = {}
    with open(os.path.join(os.path.join(dataset_dir, 'train'),
                           'relational_context.csv'),
              'r') as f:
        train_data = f.readlines()
        train_data.pop(0)
        for line in train_data:
            cols = line.strip('\n').split(',')
            node = cols[0]
            context = cols[-1]
            if node in combined:
                combined[node] = combined[node] + ";" + context
            else:
                combined[node] = context
        train_data = [x.strip('\n').split(',')[-1] for x in train_data]
        train_data_incoming = train_data[0::2]
        train_data_outgoing = train_data[1::2]
    return train_data_incoming, train_data_outgoing, combined


def get_unique_relational_contexts(contexts):
    unique_contexts = set()
    for context in contexts:
        unique, counts = torch.unique(torch.tensor([int(x) for x in
                                                    context.split()],
                                                   dtype=torch.int32),
                                      return_counts=True)
        combined = str(unique.tolist()) + ',' + str(counts.tolist())
        unique_contexts.add(combined)
    return unique_contexts


def get_unique_combined_contexts(contexts):
    unique_contexts = set()
    for key in contexts.keys():
        both_contexts = contexts[key].split(';')
        combined = ""
        for i in both_contexts:
            unique, counts = torch.unique(torch.tensor([int(x) for x in
                                                        i.split()],
                                                       dtype=torch.int32),
                                          return_counts=True)
            combined += str(unique.tolist()) + ',' + str(counts.tolist()) + ','
        combined.strip(',')
        unique_contexts.add(combined)
    return unique_contexts



def get_unique_paths(paths):
    unique_for_parts = {}
    for part in paths.keys():
        path_list = paths[part]
        unique = set([str(x) for x in path_list])
        unique_for_parts[part] = unique
    return unique_for_parts


dataset = 'wn18rr'
dataset_path = get_dataset_directory(dataset)
paths = load_paths(dataset_path)
in_context, out_context, combined_context = load_rel_context(dataset_path)
in_unique = get_unique_relational_contexts(in_context)
out_unique = get_unique_relational_contexts(out_context)
comb_unique = get_unique_combined_contexts(combined_context)
unique_paths = get_unique_paths(paths)
print()
