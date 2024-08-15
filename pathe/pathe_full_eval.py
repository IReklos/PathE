import os
import datetime
from functools import partial

import torch
import pandas as pd
from tqdm import tqdm

import triple_lib
import data_utils as du
from pather_models import PathEModel
from pathdata import EntityMultiPathDataset
from wrappers import PathEModelWrapper
from path_lib import encode_relcontext_freqs
from data_utils import collate_multipaths, load_triple_tensors
from corruption import CorruptLinkGeneratorEval, CorruptLinkGenerator
from utils import stageprint, bundle_arguments, namespace_to_dict
from pathe_ranking_metrics import EntityHitsAtK, EntityMRR, EntityMRR_debug

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

EVAL_TYPE = "Full"
TRIM_PATHS = False
NUM_NEGS = 14504
OVER_POS = False
USE_RELATION_HEAD = False
DEBUG = False

def debug_single_path(batch, trim_paths: bool=False, override_pos: bool=False):
    if trim_paths:
        ccounts = torch.cumsum(batch['ppt'], dim=0)
        ccounts_clone = ccounts.clone()
        ccounts[1:] = ccounts_clone[:-1]
        ccounts[0] = 0
        new_ppt = torch.ones_like(batch["ppt"])
        new_ent_paths = batch['net_input']['ent_paths'][ccounts, :]
        new_rel_paths = batch['net_input']['rel_paths'][ccounts, :]
        new_ent_lengths = batch['net_input']['ent_lengths'][ccounts]
        new_head_idxs = batch['net_input']['head_idxs'][ccounts]
        new_pos = batch['net_input']['pos'][ccounts, :]

        batch['ppt'] = new_ppt
        batch['net_input']['ent_paths'] = new_ent_paths
        batch['net_input']['rel_paths'] = new_rel_paths
        batch['net_input']['ent_lengths'] = new_ent_lengths
        batch['net_input']['head_idxs'] = new_head_idxs
        batch['net_input']['pos'] = new_pos

    if override_pos:
        device = batch['net_input']['pos'].device
        pos_size = batch['net_input']['pos'].size()
        new_pos = torch.arange(1, pos_size[1]+1, device=device).repeat(
            pos_size[0], 1)
        new_pos[(batch['net_input']['pos'] == 0)] = 0
        previous = batch['net_input']['pos']
        batch['net_input']['pos'] = new_pos
        del(previous)


    return batch



def run_full_eval(args):
    """
    A function that evaluates a model on the test set
    """

    stageprint("Running " + EVAL_TYPE + " evaluation.")
    with (((torch.no_grad()))):

        start_time = datetime.datetime.now()
        stageprint(
            f"Starting {args.expname} at: {start_time.strftime('%H:%M:%S')}")
        dataset = args.train_paths.split('/')[-3]
        # model_name = args.checkpoint.split('/')[-4]
        model_name = dataset

        stageprint("Creating entity dataset and dataloader")

        train_triples, val_triples, test_triples = load_triple_tensors(
            args.train_paths, args.valid_paths, args.test_paths)
        paths, relcon, _ = du.load_unrolled_setup(args.train_paths,
                                                  args.path_setup)

        # Creating filtration dictionaries and utilities for link prediction
        filtration_dict = triple_lib.make_relation_filter_dict_no_sp_tokens(
            train_triples, val_triples, test_triples)

        # Creating the head and tail filtered dict for the triple corruptor
        head_filter_dict, tail_filter_dict = triple_lib.make_head_tail_dicts(
            train_triples, val_triples, test_triples)
        unique_entities = triple_lib.get_unique_entities(
            train_triples, val_triples, test_triples)

        ppe = args.max_ppt // 2

        # To switch between conbined paths and simplepaths
        # go to pathdata.py and comment/uncomment line 1180
        # return self._getitem_combined(index)

        entity_dataset = EntityMultiPathDataset(
            path_store=paths,
            relcontext_store=relcon,
            triple_store=train_triples,
            unique_entities=unique_entities,
            paths_per_entity=ppe,
            seed=args.seed,
        )
        collate_fn = partial(
            collate_multipaths,
            padding_idx=entity_dataset.tokens_to_idxs["PAD"])
        dataloader = torch.utils.data.DataLoader(
            entity_dataset, batch_size=args.val_batch_size,
            collate_fn=collate_fn, shuffle=False,
            pin_memory=False, num_workers=args.num_workers)

        stageprint("Creating model and loading checkpoints")
        # getting number of unique entities
        num_entities = unique_entities.size()[0]
        # This should assume model and basic parameters
        relcontext_graph = encode_relcontext_freqs(
            relcontext=relcon,
            num_entities=num_entities,
            num_relations=entity_dataset.vocab_size - 2,
            offset=2,  # applied to both ents and rels
        )
        bundle = partial(bundle_arguments, exclude=["vocab_size"],
                         args=namespace_to_dict(args))
        model = PathEModel(
            vocab_size=entity_dataset.vocab_size,
            padding_idx=entity_dataset.tokens_to_idxs["PAD"],
            relcontext_graph=relcontext_graph,
            **bundle(target_class=PathEModel),
        )
        args_dict = vars(args).copy()
        args_dict.pop("checkpoint")

        # XXX Loading the model and overwriting waits with the checkpoint
        pl_model = PathEModelWrapper.load_from_checkpoint(
            **args_dict,
            checkpoint_path=args.checkpoint,
            pathe_model=model,
            filtration_dict=filtration_dict,  # FIXME# model hparameters
        )

        model = pl_model.model
        model.to(device="cuda")
        model.eval()
        stageprint("Saving positional encodings")
        torch.save(model.pos_embeddings.weight.data, model_name + "_pos.pt")

        stageprint("Initialising evaluation metrics")

        triple_corruptor = CorruptLinkGeneratorEval(
            head_filter_dict=head_filter_dict,
            tail_filter_dict=tail_filter_dict,
            entities=unique_entities, num_tensor=10)

        linkHitsAt1 = EntityHitsAtK(k=1)
        linkHitsAt3 = EntityHitsAtK(k=3)
        linkHitsAt5 = EntityHitsAtK(k=5)
        linkHitsAt10 = EntityHitsAtK(k=10)
        if DEBUG:
            MRR = EntityMRR_debug()
            test_triples = torch.load(
                "../experiments/" + dataset + "/" + str(
                    NUM_NEGS) + "_testtr.pt")
            test_triples = test_triples.reshape(test_triples.size()[
                                                    0] // ((2 * NUM_NEGS) + 2),
                                                (2 * NUM_NEGS) + 2, 3)
        else:
            MRR = EntityMRR()

        stageprint("Creating Embedding Table")

        if args.ent_aggregation == "avg":
            embedding_table = torch.zeros((unique_entities.size()[0],
                                           args.d_model), dtype=torch.float32)
        else:
            embedding_list = []
        data_iterator = iter(dataloader)
        # for each batch of entities pass them through the encoder of the model
        # and store their embeddings in the embedding table
        for batch in tqdm(data_iterator):
            if not TRIM_PATHS and not OVER_POS:
                ent_paths, rel_paths = \
                    batch["net_input"]["ent_paths"].cuda(), batch["net_input"][
                        "rel_paths"].cuda()
                indices = batch["id"].cuda()
                idxs = batch["net_input"]["head_idxs"].cuda()
                pos = batch["net_input"]["pos"].cuda()
                idxs = idxs * 2
                ppe = batch["ppt"].cuda()
                if args.ent_aggregation == "avg":
                    embedding_table[indices] = model.embed_nodes(ent_paths,
                                                                 rel_paths,
                                                                 idxs,
                                                                 pos,
                                                                 ppe,
                                                                 ).cpu()
                else:
                    embedding_list.extend(model.embed_nodes(ent_paths,
                                                            rel_paths,
                                                            idxs,
                                                            pos,
                                                            ppe,
                                                            ))
            else:
                batch = debug_single_path(batch, TRIM_PATHS, OVER_POS)
                ent_paths, rel_paths = \
                    batch["net_input"]["ent_paths"].cuda(), batch["net_input"][
                        "rel_paths"].cuda()
                indices = batch["id"].cuda()
                idxs = batch["net_input"]["head_idxs"].cuda()
                pos = batch["net_input"]["pos"].cuda()
                idxs = idxs * 2
                ppe = batch["ppt"].cuda()
                if args.ent_aggregation == "avg":
                    embedding_table[indices] = model.embed_nodes(ent_paths,
                                                                 rel_paths,
                                                                 idxs,
                                                                 pos,
                                                                 ppe,
                                                                 ).cpu()
                else:
                    embedding_list.extend(model.embed_nodes(ent_paths,
                                                            rel_paths,
                                                            idxs,
                                                            pos,
                                                            ppe,
                                                            ))



        if args.ent_aggregation != "avg":
            embedding_table = torch.nn.utils.rnn.pad_sequence(embedding_list,
                                                              batch_first=True)

        stageprint("Evaluating on all nodes")
        for i in tqdm(range(test_triples.size()[0])):
            k = NUM_NEGS # the number of negatives to sample per triple
            # Get batch of head and tail corruptions for the first triple
            if EVAL_TYPE == 'Partial':
                if not DEBUG:
                    test_batch = (
                        triple_corruptor.get_filtered_corrupted_triples_set(
                            test_triples[i].unsqueeze(0), k)).squeeze()
                    test_batch = test_batch.reshape(test_batch.size()[
                                                        0] * test_batch.size()[1], 3)
                else:
                    test_batch = test_triples[i]
                # Get the embeddings of the entities from the table
                head_embed = embedding_table[test_batch[:, 0]].cuda()
                tail_embed = embedding_table[test_batch[:, 2]].cuda()
                relation = test_batch[:, 1]

                # get the scores for the triples
                if not USE_RELATION_HEAD:
                    scores = model.predict_links(head_embed, tail_embed,
                                             relation.cuda()).squeeze().cpu()
                else:
                    scores = model.predict_relations(head_embed,
                                                     tail_embed).squeeze().cpu()
                    # logits = model.predict_relations(head_embed,
                    #                                  tail_embed).squeeze().cpu()
                    # scores = logits[torch.arange(logits.size()[0]),
                    #                 relation]

                # update the metrics
                linkHitsAt1.update(test_batch, scores, k)
                linkHitsAt3.update(test_batch, scores, k)
                linkHitsAt5.update(test_batch, scores, k)
                linkHitsAt10.update(test_batch, scores, k)
                MRR.update(test_batch, scores, k)
            elif EVAL_TYPE == 'Full':
                counts, test_list =\
                    triple_corruptor.get_filtered_corrupted_triples_and_count(
                     test_triples[i].unsqueeze(0))
                # test_batch = torch.cat(test_list,dim=1).squeeze()
                for j in range(len(test_list)):
                    test_batch = test_list[j].squeeze()
                    # Get the embeddings of the entities from the table
                    head_embed = embedding_table[test_batch[:, 0]].cuda()
                    tail_embed = embedding_table[test_batch[:, 2]].cuda()
                    relation = test_batch[:, 1]

                    # get the scores for the triples
                    # get the scores for the triples
                    if not USE_RELATION_HEAD:
                        scores = model.predict_links(head_embed, tail_embed,
                                                     relation.cuda()).squeeze().cpu()
                    else:
                        logits = model.predict_relations(head_embed,
                                                         tail_embed).squeeze().cpu()
                        scores = logits[torch.arange(logits.size()[0]),
                                        relation]
                    # update the metrics
                    linkHitsAt1.update(test_batch, scores, counts[j])
                    linkHitsAt3.update(test_batch, scores, counts[j])
                    linkHitsAt5.update(test_batch, scores, counts[j])
                    linkHitsAt10.update(test_batch, scores, counts[j])
                    MRR.update(test_batch, scores, counts[j])
            else:
                print("Unknown evaluation type, please select either Full of "
                      "Partial.")

        # calculate the metrics over the entire test set
        h1 = linkHitsAt1.compute()
        h3 = linkHitsAt3.compute()
        h5 = linkHitsAt5.compute()
        h10 = linkHitsAt10.compute()
        mrr = MRR.compute()


        # print and save them
        res_dict = {"dataset": dataset,
                    "num_negs": NUM_NEGS if EVAL_TYPE != "Full" else 0,
                    "model": model_name,
                    "wandb": " ",
                    "hits1": h1.item(),
                    "hits3": h3.item(),
                    "hits5": h5.item(),
                    "hits10": h10.item(),
                    "mrr": mrr.item(),
                    }
        stageprint("Results")
        print(res_dict)
        if os.path.exists(os.path.join(os.getcwd(), dataset + ".csv")):
            res = pd.read_csv(os.path.join(os.getcwd(), dataset + ".csv"))
            res = res._append(res_dict, ignore_index=True)
            res.to_csv(os.path.join(os.getcwd(), dataset + ".csv"), index=False)
        else:
            res = pd.DataFrame([res_dict])
            res.to_csv(os.path.join(os.getcwd(), dataset + ".csv"), index=False)
