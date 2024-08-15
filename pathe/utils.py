"""
General-purpose Python utilities.
"""
import os
import csv
import glob
import json
import random
import inspect
import logging
import argparse
from argparse import Namespace
from typing import Union, Dict, List

import torch
import numpy as np
from itertools import groupby


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def stageprint(text, logger=None, symbol="*"):
    sep = symbol * len(text)
    f = print if logger is None else logger.info
    f(f"\n{sep}\n{text}\n{sep}")


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def remove_consecutive_repeats(t):
    return [x[0] for x in groupby(t)]


def sample_or_repeat(alist: list, size: int):
    """Samples `size` unique elements from the list or repeats until `size`."""
    if len(alist) == 0:
        return [None] * size
    list_indexes = np.arange(len(alist))
    list_indexes = np.resize(list_indexes, size) if size > len(alist) \
        else np.random.choice(list_indexes, size, replace=False) 
    return [alist[i] for i in list_indexes]


def is_file(parser, f_arg):
    if not os.path.exists(f_arg):
        return parser.error("File %s does not exist!" % f_arg)
    return f_arg  # returned only if the file exists


def set_logger(log_name, log_console=True, log_dir=None):
    
    logger_master = logging.getLogger(log_name)
    logger_master.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger_master.addHandler(ch)

    if log_dir:
        fh = logging.FileHandler(
            os.path.join(log_dir,
            f'{log_name}.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger_master.addHandler(fh)

    return logger_master


def set_random_seed(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_type = device if isinstance(device, str) else device.type
    if device_type in ['cuda', 'gpu']:  # seeding torch if using GPU(s)
        torch.cuda.manual_seed_all(seed)
        # torch.cuda.set_device(args.device)


def write_csv(results_dict: str, out_fname: str):
    with open(out_fname, 'w') as f:  # 'wb' if in Python 2.x
        w = csv.DictWriter(f, results_dict.keys())
        w.writeheader()  # dict keys are the header
        w.writerow(results_dict)


def namespace_to_dict(namespace):
    """
    Convert a Namespace into a dictionary, recursively.
    """
    return {
        k: namespace_to_dict(v) if isinstance(v, Namespace) else v
        for k, v in vars(namespace).items()
    }


def bundle_arguments(args: Union[Dict, Namespace], 
                     target_class, exclude: List[str] = []):
    """
    Filter out and bundle the given input parameters based on the costructor
    of a target class; this also exclude the parameters in the black list. 

    Parameters
    ----------
    args : Union[Dict, Namespace]
        The input parameter specification given as key-value pairs. This can
        be either a dictionary or a (argparse) namespace for convenience.
    target_class : class
        The target class whose constructor parameters define the filtration. 
    exclude : list
        An optional list of parameter names to exclude from the filtration.

    Note: if the target class also contains **kwargs then this is easier
    """
    args = namespace_to_dict(args) if isinstance(args, Namespace) else args
    cparams = inspect.signature(target_class.__init__)
    parameters = [c for c in cparams.parameters
                if c not in ["self"] + exclude]
    # Create a dictionary where only the param selection is allowed
    filtered_kwa = {k: v for k, v in args.items() if k in parameters}

    return filtered_kwa


def resume_logging_setup(log_dir, exp_name, version=None, wandb_project=None):
    """
    Retrieves logs from a trained run and performs sanity check of the setup.

    Parameters
    ----------
    log_dir : str
        The path of the main log directory: the entry point to all logs.
    exp_name : str
        The name of this experiment, which identifies the related subfolder.
    version : int, optional
        The version of this experiment/run to be resumed, by default None
    wandb_project : str, optional
        If wandb is enabled, this corresponds to the project, by default None.

    Returns
    -------
    dict
        A dictionary providing pointers to logging and checkpointing.

    Raises
    ------
    ValueError
        If the setup is not consistent or cannot be retrieved.

    """
    exp_dir = os.path.join(log_dir, exp_name)
    if not os.path.isdir(exp_dir):  # sanity chek if log dir exists
        raise ValueError(f"Folder {log_dir} has no data for {exp_name}")

    versions = [int(v.split("_")[1]) for v in os.listdir(exp_dir)
                if v.startswith("version_")]  # all versions

    if len(versions) == 0:  # cannot resume a run that was not started/logged
        raise ValueError("No version of this experiment exists to be resumed")
    elif version is None and len(versions) > 1:  # cannot choose 1 version
        raise ValueError("Cannot infer exp version as multiple were found")
    elif version is None:  # can infer the version as only 1 is there
        version = versions[0]
    elif version not in versions:  # selected version is not available
        raise ValueError(f"Version {version} was not found")

    exp_dir = os.path.join(exp_dir, f"version_{version}")

    # Step 2: check if there is a valid checkpoint in the exp folder
    ckp_dir = os.path.join(exp_dir, "checkpoints")
    checkpoint = glob.glob(os.path.join(ckp_dir, "*.ckpt"))
    if len(checkpoint) == 0:
        raise ValueError(f"No checkpoint in {ckp_dir}")
    elif len(checkpoint) > 1:
        raise ValueError(f"Too many checkpoints in {ckp_dir}")
    checkpoint = checkpoint[0]  # safe to assume 1 here
    
    # Step 3: check if there is a wandb project to resume
    if wandb_project is not None:
        wandb_dir = os.path.join(exp_dir, "wandb")
        if not os.path.isdir(wandb_dir):  # no wandb folder
            raise ValueError(f"No wandb logs were found in {exp_dir}. "
                              "You can still resume training with local "
                              "logging by disabling --wandb_project")

        wandb_runs = [v.split("-")[-1] for v in os.listdir(wandb_dir)
                      if v.startswith("run-")]
        if len(wandb_runs) == 0 or len(wandb_runs) > 1:
            raise ValueError("Inconsistent setup configuration")
        
        wandb_id = wandb_runs[-1]

    return {
        "exp_dir": exp_dir,
        "version": version,
        "checkpoint": checkpoint,
        "checkpoint_dir": ckp_dir,
        "wandb_id": wandb_id if wandb_project is not None else None,
    }


def update_args(args, mapping):
    for item, value in mapping.items():
        setattr(args, item, value)


def create_logging_setup(log_dir, exp_name, version=None):
    """
    Create a new experimental setup for logging and checkpointing.

    Parameters
    ----------
    log_dir : str
        The path of the main log directory: the entry point to all logs.
    exp_name : str
        The name of this experiment, which identifies the related subfolder.
    version : int, optional
        The version of this experiment/run to be created, by default None.

    Returns
    -------
    dict
        A dictionary holding pointers to logging resources.

    Raises
    ------
    ValueError
        If attempting to create a versioned-run that already exists.
    
    """
    exp_dir = create_dir(os.path.join(log_dir, exp_name))
    versions = [int(v.split("_")[1]) for v in os.listdir(exp_dir)
                if v.startswith("version_")]  # all versions 

    if version in versions:  # avoid over-writing logs
        raise ValueError(f"Run version {version} for {exp_name} already exists!"
                          " Either resume this run, or create a new one.")
    version = max(versions + [-1]) + 1 if version is None else version
    exp_dir = create_dir(os.path.join(exp_dir, f"version_{version}"))
    # And finally we create the checkpoint folder inside 
    ckp_dir = create_dir(os.path.join(exp_dir, "checkpoints"))

    return {
        "exp_dir": exp_dir,
        "version": version,
        "checkpoint_dir": ckp_dir,
        "wandb_id": None,
    }


def dump_configuration(args: argparse.ArgumentParser, logdir: str):
    """Dump a given argparse as a json file."""
    with open(os.path.join(logdir, "experiment_setup.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_configuration(logdir: str):
    """Load the argparse from a config file expected in the given directory."""
    config_path = os.path.join(logdir, "experiment_setup.json")
    if not os.path.isfile(config_path):  # sanity check
        raise ValueError(f"No configuration file in {logdir}")

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(config_path, 'r') as f:
        args.__dict__ = json.load(f)

    return args
