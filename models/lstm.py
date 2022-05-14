#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unicodedata import bidirectional

import torch.nn as nn
from configuration import lstm_hidden_size


class LSTM(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, vectors):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_size*2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # todo check batch first 是否正确
        x = self.embedding(x)
        output, _ = self.lstm(x)
        x = self.fc(output[:, -1, :])
        x = self.sigmoid(x)
        return x
