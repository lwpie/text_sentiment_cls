#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
from cgi import test
from distutils import text_file
from pkgutil import get_data

import torchtext.data as data
from gensim.models.keyedvectors import KeyedVectors
from matplotlib.pyplot import text
from sklearn.preprocessing import binarize
from sklearn.utils import shuffle
from torchtext.vocab import Vectors


def bin2txt(data_dir):
    if os.path.exists(os.path.join(data_dir, 'wiki_word2vec_50.txt')):
        print('TXT version of word2vec file already exists.')
    else:
        print('Generating txt eord2vec file')
        vecs = KeyedVectors.load_word2vec_format(
            os.path.join(data_dir, 'wiki_word2vec_50.bin'), binary=True)
        vecs.save_word2vec_format(os.path.join(
            data_dir, 'wiki_word2vec_50.txt'), binary=False)
        print('done')


def get_tsv(file_dir):
    if (os.path.exists('data/train.tsv') and os.path.exists('data/test.tsv') and os.path.exists('data/validation.tsv')):
        print('TSV files already exists.')
        return
    print('Building tsv files..')
    train_file_pth = os.path.join(file_dir, 'train.txt')
    test_file_pth = os.path.join(file_dir, 'test.txt')
    val_file_pth = os.path.join(file_dir, 'validation.txt')

    assert os.path.exists(train_file_pth) and os.path.exists(
        test_file_pth) and os.path.exists(val_file_pth)

    modes = ['train', 'test', 'validation']

    for mode in modes:
        file_pth = os.path.join(file_dir, f'{mode}.txt')
        assert(os.path.exists(file_pth))
        with open(f'data/{mode}.tsv', 'w+', encoding="utf8")as t:
            tsv_w = csv.writer(t, delimiter='\t')
            tsv_w.writerow(['label', 'text'])
            with open(file_pth, 'r', encoding='utf8')as f:
                for line in f.readlines():
                    # 去掉str左右端的空格并以空格分割成list
                    line_list = line.strip('\n').split()
                    tsv_w.writerow([line_list[0], line_list[1:]])
    print('done')


def get_dataset(args):
    def remain(txt):
        return txt
    text_field = data.Field(lower=True)  # 文本域
    label_field = data.Field(sequential=False)  # 标签域
    text_field.tokenize = remain
    train_dataset, val_dataset = data.TabularDataset.splits(
        path='data',
        format='tsv',
        train='train.tsv',
        validation='validation.tsv',
        fields=[('label', label_field), ('text', text_field)],
        skip_header=True,
    )
    vectors = Vectors(os.path.join(
        args.data_dir, 'wiki_word2vec_50.txt'), 'data')
    text_field.build_vocab(train_dataset, val_dataset, vectors=vectors)
    label_field.build_vocab(train_dataset, val_dataset)
    print(text_field.vocab.vectors)
    return train_dataset, val_dataset


def get_dataloader(args, text_field, label_field):

    train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
        path='data',
        format='tsv',
        train='train.tsv',
        validation='validation.tsv',
        test='test.tsv',
        fields=[('label', label_field), ('txt', text_field)],
        skip_header=True,
    )

    vectors = Vectors(os.path.join(
        args.data_dir, 'wiki_word2vec_50.txt'), 'data')
    # print(vectors)
    # quit()
    text_field.build_vocab(train_dataset, val_dataset,
                           test_dataset, vectors=vectors)
    label_field.build_vocab(train_dataset, val_dataset)
    # print(text_field.vocab.vectors)
    # quit()

    # print(text_field.vocab.itos[57])
    # print(text_field.vocab.itos[16917])
    # print(text_field.vocab.itos[32761])
    # print(text_field.vocab.itos[35534])

    print(f'LOADING training set with size{len(train_dataset)}')
    print(f'LOADING validation set with size{len(val_dataset)}')
    print(f'LOADING test set with size{len(test_dataset)}')

    train_itr, val_itr, test_itr = data.Iterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_sizes=(args.batch_size, len(val_dataset), len(test_dataset)),
        shuffle=False,
        repeat=False,
        # sort_key=lambda x: len(x.text),
    )  # 构造迭代器
    return train_itr, val_itr, test_itr, text_field.vocab.vectors


if __name__ == "__main__":
    get_tsv('/data/ovo/text_sentiment_data')
    bin2txt('/data/ovo/text_sentiment_data')
    train_dataset, val_dataset = get_dataset()
    print(len(train_dataset))
    print(len(val_dataset))
    # print(train_dataset[0].label,train_dataset[0].text)
    # print(train_dataset[1].label,train_dataset[1].text)
