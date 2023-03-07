from typing import Dict

import torch
from torch import nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: spatial dropout
        self.encoder = nn.LSTM(
            embeddings.size(1), hidden_size,
            batch_first=True, num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional)
        self.enc_size = hidden_size * (2 if bidirectional else 1)
        self.pooled_size = self.enc_size * 3  # last+avg+maxpool
        self.head = nn.Sequential(
            nn.Linear(self.pooled_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_class),
        )

    @property
    def encoder_output_size(self) -> int:
        return self.enc_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.embed(batch['token_ids'])  # (bs, seq, embed_size)
        h, c = self.encoder(x)  # h = (bs, seq, enc_size)

        # pool last state, avg and max pool of hidden states to fc layer
        h_last = h[:, -1, :]
        h_avg = h.mean(1)
        h_max, _ = h.max(1)
        x = torch.cat([h_last, h_avg, h_max], 1)  # (bs, pooled_size)

        return self.head(x)  # (bs, num_class)


class SeqTagger(SeqClassifier):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.encoder = nn.LSTM(
            embeddings.size(1), hidden_size,
            batch_first=True, num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional)
        self.enc_size = hidden_size * (2 if bidirectional else 1)
        self.pooled_size = self.enc_size  # last+avg+maxpool
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.pooled_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.embed(batch['token_ids'])  # (bs, seq, embed_size)
        # TODO: spatial dropout; double check that we are dropping embedding channels, not whole tokens
        # https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/4
        h, c = self.encoder(x)  # h = (bs, seq, enc_size)
        return self.head(h)  # (bs, seq, num_class)
