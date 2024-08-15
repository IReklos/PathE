"""
Scalable routines for training a PathE model.

"""
import os
import logging
import datetime
from functools import partial

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

import triple_lib
from pather_models import PathEModel
from pathdata import NegativeTripleSampler, TripleEntityMultiPathDataset
from data_utils import collate_multipaths, load_triple_tensors, \
    load_corrupted_triples_from_dir, memmap_corrupted_triples_from_dir
from callbacks import DatasetUpdater
from corruption import CorruptHeadGenerator, CorruptTailGenerator, \
    CorruptLinkGenerator
from utils import stageprint, bundle_arguments, namespace_to_dict
from wrappers import PathEModelWrapper
from path_lib import encode_relcontext_freqs
import data_utils as du

logger = logging.getLogger(__name__)


def create_and_run_training_exp(args):
    """
    Main entry point for the experimental setup, execution, and evaluation.
    """
    model_name = args.model
    start_time = datetime.datetime.now()
    stageprint(f"Starting {args.expname} at: {start_time.strftime('%H:%M:%S')}")

    stageprint("Loading data and creating corruptions...")
    # Case A: [UnrolledPathDataset] only for loading
    train_triples, val_triples, test_triples = load_triple_tensors(
        args.train_paths, args.valid_paths, args.test_paths)
    paths, relcon, _ = du.load_unrolled_setup(args.train_paths, args.path_setup)

    # Creating filtration dictionaries and utilities for link prediction
    filtration_dict = triple_lib.make_relation_filter_dict_no_sp_tokens(
        train_triples, val_triples, test_triples)
    # Creating the head and tail filtered dict for the triple corruptors
    head_filter_dict, tail_filter_dict = triple_lib.make_head_tail_dicts(
        train_triples, val_triples, test_triples)
    unique_entities = triple_lib.get_unique_entities(
        train_triples, val_triples, test_triples)
    class_weights = triple_lib.get_class_weights_without_special_tokens(
        train_triples) if args.class_weigths else None

    # Creating the triple corruptors for head, tail, or both (merged) and
    # preserving the number of positive triples in case of H/T corruption
    tr_positives, va_positives, te_positives = \
        len(train_triples), len(val_triples), len(test_triples)
    triple_corruptor = None  # instantiated only if num_negatives > 0
    if args.num_negatives > 0 and args.corruption in ["h", "t"]:
        triple_corruptor = CorruptHeadGenerator(
            filter_dict=head_filter_dict, entities=unique_entities) \
            if args.corruption == "h" else CorruptTailGenerator(
            filter_dict=tail_filter_dict, entities=unique_entities)
    elif args.num_negatives > 0:  # "all" applies and merge both corruptions
        triple_corruptor = CorruptLinkGenerator(
            head_filter_dict=head_filter_dict,
            tail_filter_dict=tail_filter_dict,
            entities=unique_entities)
        # The number of positive is simply duplicated to measure LP results
        tr_positives, va_positives, te_positives = \
            [i * 2 for i in [tr_positives, va_positives, te_positives]]

    parallel = True
    negatives = [None, None, None]  # assume no negative dumpset is available
    if args.dump_dir is not None:
        print(f"Loading checkpointed negative triples from {args.dump_dir}")
        # The training pos-neg triples are loaded as a numpy memory map 
        negatives[0] = memmap_corrupted_triples_from_dir(
            os.path.join(args.dump_dir, "train"))
        # Validation and test triples are simply loaded as torch tensors
        negatives[1] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "valid"))
        negatives[2] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "test"))

    stageprint("Creating datasets and dataloaders...")
    train_set = TripleEntityMultiPathDataset(
        path_store=paths,
        relcontext_store=relcon,
        triple_store=train_triples,
        context_triple_store=train_triples,
        maximum_triple_paths=args.max_ppt,
        num_negatives=args.num_negatives,
        triple_corruptor=triple_corruptor,
        parallel=parallel,
        neg_triple_store=negatives[0],
    )
    # Using shared data structures for valid and test
    tokens_to_idxs = train_set.tokens_to_idxs
    path_store = train_set.relation_paths, \
                 train_set.entity_paths, \
                 train_set.path_index
    valid_set = TripleEntityMultiPathDataset(
        path_store=path_store,
        relcontext_store=relcon,
        triple_store=val_triples,
        context_triple_store=train_triples,
        maximum_triple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        triple_corruptor=triple_corruptor,
        parallel=parallel,
        neg_triple_store=negatives[1],
    )
    test_set = TripleEntityMultiPathDataset(
        path_store=path_store,
        relcontext_store=relcon,
        triple_store=test_triples,
        context_triple_store=train_triples,
        maximum_triple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        triple_corruptor=triple_corruptor,
        parallel=parallel,
        neg_triple_store=negatives[2],
    )

    if args.num_negatives > 0 and args.dump_negatives:
        print(f"Dumping negative triples in {args.dump_dir}")
        for n, d in zip(["tr", "va", "te"], [train_set, valid_set, test_set]):
            negt = d.triplestore  # FIXME via the dump_negatives()
            torch.save(negt, os.path.join(args.dump_dir, f'{n}_negatives.pt'))

    print("Found {} samples in the dataset: Tr {}, Va {}, Te {}"
          .format(len(train_set) + len(valid_set) + len(test_set),
                  len(train_set), len(valid_set), len(test_set)))
    print(f"Vocabulary size (with special tokens): {len(tokens_to_idxs)}")

    if args.val_batch_size % (args.val_num_negatives + 1) != 0:
        positive_batches = args.val_batch_size // (args.val_num_negatives + 1)
        args.val_batch_size = positive_batches * (args.val_num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping val/test batch size to:"
                       f" {args.val_batch_size}")
    if args.batch_size % (args.num_negatives + 1) != 0:
        positive_batches = args.batch_size // (args.num_negatives + 1)
        args.batch_size = positive_batches * (args.num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping train batch size to: {args.batch_size}")

    # Creating the data loaders for each partition
    collate_fn = partial(collate_multipaths, padding_idx=tokens_to_idxs["PAD"])
    tr_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=False, num_workers=args.num_workers,
        sampler=NegativeTripleSampler(tr_positives, args.num_negatives))
    va_dataloader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=False, num_workers=args.num_workers,
        sampler=NegativeTripleSampler(va_positives, args.val_num_negatives))
    te_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=False, num_workers=args.num_workers,
        sampler=NegativeTripleSampler(te_positives, args.val_num_negatives))

    stageprint("Creating model and loading checkpoints")
    # getting number of unique entities
    num_entities = unique_entities.size()[0]
    # This should assume model and basic parameters
    relcontext_graph = encode_relcontext_freqs(
        relcontext=relcon,
        num_entities=num_entities,
        num_relations=train_set.vocab_size - 2,
        offset=2,  # applied to both ents and rels
    )
    bundle = partial(bundle_arguments, exclude=["vocab_size"],
                     args=namespace_to_dict(args))
    model = PathEModel(
        vocab_size=train_set.vocab_size,
        padding_idx=tokens_to_idxs["PAD"],
        relcontext_graph=relcontext_graph,
        **bundle(target_class=PathEModel),
    )
    # class_weights = triple_lib.get_class_weights(train_triples, tokens_to_idxs)
    pl_model = PathEModelWrapper(
        pathe_model=model,
        filtration_dict=filtration_dict,  # FIXME
        class_weights=class_weights,  # for rel imbalance
        **namespace_to_dict(args),  # model hparameters
    )
    print(pl_model.model)  # keras-style model overview
    pl_model.model.to(torch.device(args.device))

    stageprint("Creating loggers, callbacks and setting up trainer")
    # Registering loggers based on the experiment name and version
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.expname,
        version=args.version)
    exp_dir = tb_logger.log_dir  # equivalent to args.exp_dir
    if args.wandb_project is not None:  # use synchronised wandb logger
        wb_logger = WandbLogger(
            id=args.wandb_id,
            save_dir=exp_dir,
            name=args.expname,
            project=args.wandb_project,
            log_model="all",
            sync_tensorboard=True)
        # Logging gradients, topology, histogram
        wb_logger.watch(pl_model, log="all")
        wb_logger.log_hyperparams(args)

    # Creating callbacks for checkpointing and early stopping
    mode = "min" if args.monitor.endswith("loss") else "max"
    checkpoint_callbk = ModelCheckpoint(
        monitor=args.monitor, dirpath=args.checkpoint_dir, mode=mode,
        filename=model_name + f"-{{epoch}}-{{{args.monitor}:.2f}}",
        every_n_train_steps=args.chekpoint_ksteps)
    estopping_callbk = EarlyStopping(
        monitor=args.monitor, patience=args.patience, mode=mode)
    dataset_callbk = DatasetUpdater(  # datasets to seed per-epoch
        [train_set, valid_set] if args.train_paths else [train_set.dataset])

    # accumulated_batches = 5 # {2: 3, 4: 6}
    tr_limit, va_limit = (.1, .2) if args.debug else (1., 1.)
    accelerator = "gpu" if args.device == "cuda" else "cpu"

    trainer = Trainer(
        max_epochs=args.max_epochs, gradient_clip_val=1.0,
        accelerator=accelerator, devices=args.num_devices, num_nodes=1,
        limit_train_batches=tr_limit, limit_val_batches=va_limit,
        logger=[tb_logger, wb_logger], log_every_n_steps=5,
        val_check_interval=args.val_check_interval,
        gradient_clip_algorithm='norm',  # CHANGED THIS
        accumulate_grad_batches=args.accumulate_gradient,
        callbacks=[estopping_callbk, checkpoint_callbk, dataset_callbk],
    )

    if args.cmd in ["train", "resume"]:
        # Train and resume are the same assuming their setup is consistent
        stageprint("Training-validating the model, be patient!")
        trainer.fit(pl_model, tr_dataloader, va_dataloader,
                    ckpt_path=args.checkpoint)  # load or None
        args.checkpoint = checkpoint_callbk.best_model_path
        print(f"Done. Best model saved in {args.checkpoint}")
        ttime = (datetime.datetime.now() - start_time).total_seconds() / 3600
        print(f"Training time: {round(ttime, 2)} hours")

    # stageprint("Evaluating the model on the validation set")
    # valid_dict = trainer.validate(
    #     dataloaders=va_dataloader,
    #     ckpt_path=checkpoint_callbk.best_model_path)[0]
    # del va_dataloader, valid_set  # free some memory
    # print("\nValidation results: {}".format(valid_dict))

    stageprint("Evaluating the model on the test set")
    test_dict = trainer.test(model=pl_model if args.cmd == "test" else None,
                             dataloaders=te_dataloader,
                             ckpt_path=args.checkpoint)[0]
    print("\nTesting results: {}".format(test_dict))

    # results_dict = {**valid_dict, **test_dict}
    # results_dict.update(namespace_to_dict(args))  # +hparams
    # fname = os.path.join(args.log_dir, "results_summary.csv")
    # write_csv(results_dict, fname)
    # results_raw = pd.read_csv(fname)

    # return results_dict, results_raw
