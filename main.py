import argparse
import os
import torchtext.data as data
from dataloader import get_dataloader
import torch
from models.cnn import CNN
from tensorboardX import SummaryWriter


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

    text_field = data.Field(lower=True)  # 文本域
    label_field = data.Field(sequential=False)  # 标签域

    train_dataloader, val_dataloader, test_dataloader, vectors = get_dataloader(
        args, text_field=text_field, label_field=label_field)

    vocab_size = len(text_field.vocab)

    if args.model_type == 'cnn':
        network = CNN(args, vocab_size=vocab_size,
                      embedding_dim=128, vectors=vectors)
    else:
        raise NotImplementedError

    network = network.to(device)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensor_writer'))

    # todo 加载models
    for epoch in range(args.epochs):
        train(train_dataloader, network, device,writer)
        val(val_dataloader,network,writer)
        quit()


def train(train_dataloader, network, device,writer):
    network.train()
    for i, data_all in enumerate(train_dataloader):
        txt = data_all.txt
        txt = txt.to(device)
        label = data_all.label
        label = label.to(device)

        label = (label-1).float()
        pred = network(txt)
        quit()


def val(val_dataloader,network,device,writer):
    network.eval()
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str,
                        default='/data/ovo/text_sentiment_data')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'attention'])

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    args = parser.parse_args()

    main(args)
