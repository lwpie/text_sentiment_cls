#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from configuration import transformer_hidden_size


class Transformer(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, vectors):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.W_q = nn.Linear(embedding_dim, transformer_hidden_size)
        self.W_k = nn.Linear(embedding_dim, transformer_hidden_size)
        self.W_v = nn.Linear()

    def forward(self, x):
        raise NotImplementedError
