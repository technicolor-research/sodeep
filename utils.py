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

import torch


def get_rank(batch_score, dim=0):
    rank = torch.argsort(batch_score, dim=dim)
    rank = torch.argsort(rank, dim=dim)
    rank = (rank * -1) + batch_score.size(dim)
    rank = rank.float()
    rank = rank / batch_score.size(dim)

    return rank


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def save_checkpoint(state, is_best, model_name, epoch):
    if is_best:
        torch.save(state, './weights/best_' + model_name + ".pth.tar")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_epoch(logger, epoch, train_loss, val_loss, lr, batch_train, batch_val, data_train, data_val):
    logger.add_scalar('Loss/Train', train_loss, epoch)
    logger.add_scalar('Loss/Val', val_loss, epoch)
    logger.add_scalar('Learning/Rate', lr, epoch)
    logger.add_scalar('Learning/Overfitting', val_loss / train_loss, epoch)
    logger.add_scalar('Time/Train/Batch Processing', batch_train, epoch)
    logger.add_scalar('Time/Val/Batch Processing', batch_val, epoch)
    logger.add_scalar('Time/Train/Data loading', data_train, epoch)
    logger.add_scalar('Time/Val/Data loading', data_val, epoch)


def flatten(l):
    return [item for sublist in l for item in sublist]
