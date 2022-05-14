#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from multiprocessing.sharedctypes import Value

import torch
import torch.nn as nn
import torchtext.data as data
from tensorboardX import SummaryWriter

from dataloader import get_dataloader
from models.attention import Transformer
from models.cnn import CNN
from models.loss import LOSS
from models.lstm import LSTM

train_step = 0


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # record args in configuration
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        for elem in vars(args):
            f.write(f'{elem}---->{eval(f"args.{elem}")}\n')

    save_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")

    model_dir = os.path.join(save_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    text_field = data.Field(lower=True)  # 文本域
    label_field = data.Field(sequential=False)  # 标签域

    train_dataloader, val_dataloader, test_dataloader, vectors = get_dataloader(
        args, text_field=text_field, label_field=label_field)

    vocab_size = len(text_field.vocab)

    if args.model_type == 'cnn':
        network = CNN(args, vocab_size=vocab_size,
                      embedding_dim=128, vectors=vectors)
    elif args.model_type == 'lstm':
        network = LSTM(args, vocab_size=vocab_size,
                       embedding_dim=128, vectors=vectors)
    elif args.model_ytpe == 'attention':
        network = Transformer(args, vocab_size=vocab_size,
                              embedding_dim=128, vectors=vectors)
    else:
        raise ValueError('Invalid model type.')

    network = network.to(device)
    loss_crt = LOSS(args.loss_type).to(device)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensor_writer'))
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)  # 优化器

    # todo 加载models
    for epoch in range(args.epochs):
        print(f'epoch {epoch}/{args.epochs}')
        train(train_dataloader, network, loss_crt, optimizer, device, writer)
        val(val_dataloader, network, loss_crt,
            device, writer, epoch, args, model_dir)


def train(train_dataloader, network, loss_crt, optimizer, device, writer: SummaryWriter):
    global train_step
    network.train()
    for i, data_all in enumerate(train_dataloader):
        if i % 10 == 0:
            print(f'i...')
        txt = data_all.txt
        txt = txt.to(device)
        label = data_all.label
        label = label.to(device)

        label = (label-1).float()  # todo:check whether it's right

        optimizer.zero_grad()
        pred = network(txt)
        loss = loss_crt(label, pred)
        writer.add_scalar('loss_train', loss.item(), global_step=train_step)

        loss.backward()
        optimizer.step()


def val(val_dataloader, network: nn.Module, loss_crt, device, writer, epoch, args, model_dir):
    network.eval()
    for _, data_all in enumerate(val_dataloader):
        with torch.no_grad():
            txt = data_all.txt
            txt = txt.to(device)
            label = data_all.label
            label = label.to(device)

            label = (label-1).float()  # todo:check whether it's right

            pred = network(txt)
            loss = loss_crt(label, pred)
            writer.add_scalar('loss_val', loss.item(), global_step=epoch)
        torch.save(network.state_dict(), os.path.join(
            model_dir, f'{args.model_type}_epoch{epoch}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str,
                        default='/data/ovo/text_sentiment_data')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'attention'])
    parser.add_argument('--loss_type', default='bce', type=str)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    args = parser.parse_args()

    main(args)
