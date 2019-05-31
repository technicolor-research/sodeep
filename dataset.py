"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2019 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2019, June).
    SoDeep: A Sorting Deep Net to Learn Ranking Loss Surrogates.
    In Proceedings of CVPR

Author: Martin Engilberge
"""

import numpy as np
import torch

from random import randint
from torch.utils.data import Dataset


def get_rand_seq(seq_len, ind=None):
    if ind is None:
        type_rand = randint(0, 9)
    else:
        type_rand = int(ind)

    if type_rand == 0:
        rand_seq = np.random.rand(seq_len) * 2.0 - 1
    elif type_rand == 1:
        rand_seq = np.random.uniform(-1, 1, seq_len)
    elif type_rand == 2:
        rand_seq = np.random.standard_normal(seq_len)
    elif type_rand == 3:
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / seq_len)
    elif type_rand == 4:
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / seq_len)
        np.random.shuffle(rand_seq)
    elif type_rand == 5:
        split = randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.rand(split) * 2.0 - 1, np.random.standard_normal(seq_len - split)])
    elif type_rand == 6:
        split = randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.uniform(-1, 1, split), np.random.standard_normal(seq_len - split)])
    elif type_rand == 7:
        split = randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.rand(split) * 2.0 - 1, np.random.uniform(-1, 1, seq_len - split)])
    elif type_rand == 8:
        split = randint(1, seq_len)
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / split)
        np.random.shuffle(rand_seq)
        rand_seq = np.concatenate(
            [rand_seq, np.random.rand(seq_len - split) * 2.0 - 1])
    elif type_rand == 9:
        a = -1.0
        b = 1.0
        rand_seq = np.arange(a, b, (b - a) / seq_len)
    elif type_rand == 10:
        rand_seq = np.random.rand(seq_len) * np.random.rand() - np.random.rand()

    return rand_seq[:seq_len]


class SeqDataset(Dataset):

    def __init__(self, seq_len, nb_sample=400000, dist=None):
        self.seq_len = seq_len
        self.nb_sample = nb_sample

        self.dist = dist

    def __getitem__(self, index):
        rand_seq = get_rand_seq(self.seq_len, self.dist)
        zipp_sort_ind = zip(np.argsort(rand_seq)[::-1], range(self.seq_len))

        ranks = [((y[1] + 1) / float(self.seq_len)) for y in sorted(zipp_sort_ind, key=lambda x: x[0])]

        return torch.FloatTensor(rand_seq), torch.FloatTensor(ranks)

    def __len__(self):
        return self.nb_sample


def get_rank_single(batch_score):
        rank = torch.argsort(batch_score, dim=0)
        rank = torch.argsort(rank, dim=0)
        rank = (rank * -1) + batch_score.size(0)
        rank = rank.float()
        rank = rank / batch_score.size(0)

        return rank
