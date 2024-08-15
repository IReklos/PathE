"""
This script contains the PL callback utilities and methods used in PathE.

"""
import logging
from typing import List

from pytorch_lightning.callbacks import Callback

from pathdata import PathDataset

logger = logging.getLogger(__name__)


class DatasetUpdater(Callback):
    """
    Updates the epoch count in the given dataset(s) so that perturbations can be
    seeded accordingly, instead of triggering the same transformations.
    """
    def __init__(self, datasets:List[PathDataset], verbose=False):
        self.datasets = datasets
        self.verbose = verbose
        self.state = {"epoch": 0}

    @property
    def state_key(self) -> str:
        return f"DatasetUpdater[epoch={self.state['epoch']}]"

    def _seed_datasets(self, seed):
        for dataset in self.datasets:
            dataset.set_epoch(seed)
        if self.verbose:  # logging epoch-set operation
            logger.info(f"Epoch {seed} set for datasets")

    def setup(self, *args, **kwargs) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        self._seed_datasets(self.state['epoch'])

    def on_train_epoch_start(self, *args, **kwargs):
        self.state['epoch'] += 1  # update epoch count
        self._seed_datasets(self.state['epoch'])

    def on_fit_end(self, *args, **kwargs):
        self._seed_datasets(None)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()
