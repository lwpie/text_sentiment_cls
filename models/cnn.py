#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration import cnn_kernel_size, cnn_kernel_num


class CNN(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, vectors):
        super(CNN, self).__init__()
        self.args = args

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        # print(vectors)
        # quit()
        self.embedding = self.embedding.from_pretrained(vectors, freeze=True)

        # self.convs = nn.ModuleList([nn.Conv1d()])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(cnn_kernel_size) * cnn_kernel_num, 2)  # 全连接层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(self.embedding.weight)
        x = self.embedding(x)  # lenth, batch, emb_dim
        # print(x[:,0,:])
        # quit()
        # print(x.shape)
        # quit()
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.sigmoid(self.fc(x))
        return logits
