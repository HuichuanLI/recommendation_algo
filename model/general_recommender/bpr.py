# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 14:31
# @Author  : Li Huichuan
# @File    : bpr.py
# @Software: PyCharm
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from collections import namedtuple
import pandas as pd
import numpy as np


class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, n_users, n_items, embedding_size, dataset):
        super(BPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = embedding_size
        self.n_users = n_users
        self.n_items = n_items

        # define layers and loss
        self.user_embedding = Embedding(self.n_users, self.embedding_size)
        self.item_embedding = Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.
        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]
        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.
        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]
        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)


if __name__ == "__main__":
    from recbole.quick_start import run_recbole

    run_recbole(model='BPR', dataset='ml-100k')
