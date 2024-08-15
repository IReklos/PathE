"""
Modules for aggregating representations of arbitrary size.

"""
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def simple_aggregation(embed_heads: Tensor, embed_tails: Tensor,
                       aggregator: nn.Module):
    """
    Concatenate head and tail embeddings horizontally and perform aggregation.

    Args:
        embed_heads (Tensor): Tensor containing the embeddings of the heads.
        embed_tails (Tensor): Tensor containing the embeddings of the tails.
        aggregator (nn.Module): Aggregator module to perform the aggregation.

    Returns:
        Tensor: Tensor containing the aggregated embeddings.
    """
    all_embeddings = torch.cat([embed_heads, embed_tails], dim=1)
    all_embeddings = aggregator(all_embeddings)
    return all_embeddings


class ContextualAggregator(nn.Module):
    """
    # 1. From num_embeddings to 1 embedding vector applying stage-1 aggregation
    # 2. From 1 embedding we attend to all the other embeddings with stage-2 to yield the context vector
    # 3. Finally, we sum the stage-1 embedding vector with the context vector
    """

    def __init__(self, embedding_dim, aggregator: str, context_heads=0,
                 context_dropout=.0, **kwargs):
        super().__init__()
        if aggregator == "recurrent":
            aggregator = RecurrentAggregator(
                embedding_dim=embedding_dim, **kwargs)
        elif aggregator == "transformer":
            aggregator = TransformerAggregator(
                embedding_dim=embedding_dim, **kwargs)
        elif aggregator == "single_query":
            aggregator = SingleQueryAggregator(
                embedding_dim=embedding_dim, **kwargs)
        else:
            raise ValueError("Invalid aggregator type")
        self.aggregator = aggregator  # Stage-1 aggregator
        # Stage-2 aggregators, for the contextualisation embeddings

        self.context_heads = context_heads
        if context_heads > 0 and aggregator != "transformer":
            self.h2tails_contextualiser = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim, num_heads=context_heads,
                dropout=context_dropout, batch_first=True)
            self.t2heads_contextualiser = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim, num_heads=context_heads,
                dropout=context_dropout, batch_first=True)

    def forward(self, embed_heads: Tensor, embed_tails: Tensor):
        # Stage-1 aggregation
        heads_agg = self.aggregator(embed_heads)
        tails_agg = self.aggregator(embed_tails)
        # print(heads_agg.shape, tails_agg.shape)
        # print(embed_heads.shape, embed_tails.shape)
        if self.context_heads > 0:
            # Stage-2 contextualisation
            heads_contextualised, _ = self.h2tails_contextualiser(
                heads_agg.unsqueeze(1), embed_tails, embed_tails)
            tails_contextualised, _ = self.t2heads_contextualiser(
                tails_agg.unsqueeze(1), embed_heads, embed_heads)
            # Sum the stage-1 embeddings with the context vectors
            heads_agg = heads_agg + heads_contextualised.squeeze(1)
            tails_agg = tails_agg + tails_contextualised.squeeze(1)

        return heads_agg, tails_agg


class RecurrentAggregator(nn.Module):
    """
    Implements an encoder as a bidirectional reccurrent network to agrgegate a
    sequence of embedding vectors into a single representation.
    """

    def __init__(self, embedding_dim, num_agg_layers=1, **kwargs):

        super().__init__()
        self.embed_dim = embedding_dim
        self.recurrent_net = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=embedding_dim,
                                     num_layers=num_agg_layers,
                                     batch_first=True,
                                     bidirectional=True,
                                     dropout=0.0)

    def forward(self, sequence_embed: Union[list, Tensor]) -> Tensor:
        if isinstance(sequence_embed, list):  # listed batch of tensors
            sequence_lens = [len(s) for s in sequence_embed]
            sequence_embed = pad_sequence(sequence_embed, batch_first=True)
        else:  # single tensor with padding
            sequence_lens = (torch.abs(sequence_embed).sum(dim=-1) != 0).sum(
                dim=-1).to('cpu')
        # Pack the sequence to remove padding and feed to the recurrent model
        # assert (sequence_lens == 0).nonzero().size()[0] == 0, \
        #     ("the sequence lengths cannot be zero")
        sequence_embed = pack_padded_sequence(
            sequence_embed, batch_first=True,
            enforce_sorted=False, lengths=sequence_lens)
        # Feed the sequence batch to the recurrent model and retrieve last items
        outputs, (hn, _) = self.recurrent_net(sequence_embed)
        outputs, input_sizes = pad_packed_sequence(outputs, batch_first=True)
        last_seq_idxs = torch.LongTensor([x - 1 for x in input_sizes])
        last_seq_items = outputs[range(outputs.shape[0]), last_seq_idxs, :]
        # Bididrectional outputs and hidden state are aggregated by sum
        if self.recurrent_net.bidirectional:
            aggregation = last_seq_items[:, :self.embed_dim] + \
                          last_seq_items[:, self.embed_dim:]
        else:  # Unidirectional outputs and hidden state are the same
            aggregation = last_seq_items
        # context = torch.sum(hn, dim=0)
        return aggregation


class TransformerAggregator(nn.Module):
    """
    Implements a sequence aggregator by appending an embedding vector to the
    sequence and applying N transformer encoder layers to contextually add
    information. The aggregated vector is the last embedding after the encoder.
    """

    def __init__(self, embedding_dim, num_agg_layers=2, num_agg_heads=1,
                 dropout=.1, activation=F.relu, **kwargs):
        super().__init__()
        # Uses an embedding CLS-alike vector to hold the aggregation
        self.num_agg_layers = num_agg_layers
        self.num_agg_heads = num_agg_heads
        self.agg_cls = nn.Parameter(
            torch.empty(embedding_dim), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_agg_heads,
            dim_feedforward=embedding_dim, dropout=dropout,
            activation=activation, batch_first=True)
        self.sa_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_agg_layers, norm=None)
        # Initialise the aggregation vector with normal distribution
        nn.init.normal_(self.agg_cls, mean=0.0, std=1.00)

    def device(self):
        return self.agg_cls.device

    def forward(self, sequence_embed: Tensor) -> Tensor:
        """
        This method takes a batch of sequences of embeddings and performs:
        1. Appends the aggregation vector to each sequence in the batch.
        2. Creates a mask that hides the last item (the aggregation vector).
        3. Applies the transformer encoder to the sequence, ensuring that
           padded items (0 embedding vectors) are not attended to.
        4. Returns the last item of the aggregated sequence.

        Note: The aggregation vector is obtained from the self.agg_cls attribute.

        """
        agg = self.agg_cls.reshape(1, 1, -1).repeat(sequence_embed.size(0), 1,
                                                    1)
        sequence_embed = torch.cat((sequence_embed, agg), dim=1)
        # Create padding and aggreagation mask for the transformer encoder
        mask_pad = (torch.abs(sequence_embed).sum(
            dim=-1) != 0)  # assume 0 for padding
        mask_agg = torch.ones(sequence_embed.size(0) * self.num_agg_heads,
                              sequence_embed.size(1), sequence_embed.size(1),
                              device=self.device())
        mask_agg[:, :-1, -1] = 0  # the last item is the aggregation vector
        aggregation = self.sa_encoder(sequence_embed, mask=~mask_agg.bool(),
                                      src_key_padding_mask=~mask_pad)
        return aggregation[:, -1, :]


class SingleQueryAggregator(nn.Module):
    """
    A specialised implementation of `TransformerAggregator`` when using a single
    aggregation layer and the learned aggregation embedding as a query vector.
    """

    def __init__(self, embedding_dim, num_agg_heads=1, **kwargs):
        super().__init__()
        # Uses an embedding CLS-alike vector to hold the aggregation
        self.agg_cls = nn.Parameter(
            torch.empty(embedding_dim), requires_grad=True)
        self.mha_layer = nn.MultiheadAttention(
            embedding_dim, num_heads=num_agg_heads, batch_first=True)
        # Initialise the aggregation vector with normal distribution
        nn.init.normal_(self.agg_cls, mean=0.0, std=1.00)

    def forward(self, sequence_embed: Tensor) -> Tensor:
        """
        This method takes a batch of sequences of embeddings and performs a
        multi-head attention step by using the learnable query to attend the
        given embedding sequence. It returns the attended aggregation vector.
        """
        # sequence_embed is of shape (batch_size, seq_length, embedding_dim)
        mask_pad = (sequence_embed.sum(dim=-1) != 0)  # assume 0 for padding
        # Perform MHA using self.agg_cls to attend the sequence_embed
        query = self.agg_cls.reshape(1, 1, -1).repeat(sequence_embed.size(0), 1,
                                                      1)
        aggregation, _ = self.mha_layer(query, sequence_embed, sequence_embed,
                                        key_padding_mask=~mask_pad)
        # print(aggregation.shape)
        return aggregation[:, 0, :]





# Generate some tests for all the aggregators using sample data
if __name__ == "__main__":
    # Generate some sample data
    batch_size = 3
    seq_length = 5
    embedding_dim = 4
    sequence_embed = torch.rand((batch_size, seq_length, embedding_dim))
    sequence_embed[0, 3:, :] = 0  # simulate padding
    sequence_embed[1, 4:, :] = 0  # simulate padding
    print(sequence_embed.shape)

    # Test the RecurrentAggregator
    print("\nTesting RecurrentAggregator")
    recurrent_agg = RecurrentAggregator(embedding_dim=embedding_dim)
    aggregation = recurrent_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the TransformerAggregator
    print("\nTesting TransformerAggregator")
    transformer_agg = TransformerAggregator(embedding_dim=embedding_dim)
    aggregation = transformer_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the SingleQueryAggregator
    print("\nTesting SingleQueryAggregator")
    single_query_agg = SingleQueryAggregator(embedding_dim=embedding_dim)
    aggregation = single_query_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the LAFAggregator
    print("\nTesting LAFAggregator")
    laf_agg = LAFAggregator(embedding_dim=embedding_dim, laf_units=2)
    aggregation = laf_agg(sequence_embed)
    assert aggregation.shape == (batch_size, embedding_dim), \
        "Aggregation shape does not match batch_size * embedding_dim"
    print(aggregation.shape)

    # Test the ContextualAggregator
    print("\nTesting ContextualAggregator")
    for aggregator in ["recurrent", "transformer", "single_query"]:
        print(f"Testing {aggregator} aggregator")
        contextual_agg = ContextualAggregator(embedding_dim=embedding_dim,
                                              aggregator=aggregator)
        heads_final, tails_final = contextual_agg(sequence_embed,
                                                  sequence_embed)
        assert heads_final.shape == (batch_size, embedding_dim), \
            "Aggregation shape does not match batch_size * embedding_dim"
        print(heads_final.shape)
        assert tails_final.shape == (batch_size, embedding_dim), \
            "Aggregation shape does not match batch_size * embedding_dim"
        print(tails_final.shape)
