# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 14:31
# @Author  : Li Huichuan
# @File    : bpr.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from collections import namedtuple
import pandas as pd
import numpy as np
from model.abstract_model import GeneralRecommender
from dataset import Dataset


class BPR(tf.keras.Model):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """

    def __init__(self, config, dataset):
        super().__init__()
        # load parameters info
        self.embedding_size = config['embedding_size']

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # define layers and loss
        self.user_embedding = Embedding(self.n_users, self.embedding_size)
        self.item_embedding = Embedding(self.n_items, self.embedding_size)
        self.opt = tf.keras.optimizers.Adam(0.01)

    def call(self, X, training):
        user_id = X["user_id"]
        pos_item_id = X["item_id"]
        neg_item_id = X["neg_item"]

        user_emb = self.user_embedding(user_id)
        pos_item_em = self.item_embedding(pos_item_id)
        neg_item_em = self.item_embedding(neg_item_id)

        pos_score = tf.matmul(user_emb, pos_item_em, transpose_b=True)
        neg_score = tf.matmul(user_emb, neg_item_em, transpose_b=True)
        return pos_score, neg_score

    # in order to reduce the computation of full softmax
    def loss(self, x, training=None):
        pos_score, neg_score = self.call(x, training)
        loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_score - neg_score)))
        return loss

    def calculate_loss(self, X, training=True):
        with tf.GradientTape() as tape:
            cu_loss = self.loss(X, training=True)
            grads = tape.gradient(cu_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return cu_loss.numpy()

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

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return tf.matmul(user_e, item_e, transpose_b=True)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        pass


if __name__ == "__main__":
    # 读取数据
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv", "k": 10,
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": "",
              "embedding_size": 10}

    dataset = Dataset(config=config)
    bpr = BPR(config, dataset=dataset)
    # print(history.summary())

    data = dataset.rating
    pos_data = data[data.rating > 4]
    neg_data = data[data.rating < 4]
    merge_data = pd.merge(pos_data, neg_data, how='inner', left_on='user_id', right_on='user_id')

    for t in range(10):
        for step in range(0, len(merge_data), 1000):
            X = {"user_id": np.array(merge_data["user_id"][step:step + 1000]), \
                 "item_id": np.array(merge_data["anime_id_x"][step: step + 1000]), \
                 "neg_item": np.array(merge_data["anime_id_y"][step: step + 1000])}

            loss = bpr.calculate_loss(X)
            if step % 100 == 0:
                print("step: {} | loss: {}".format(step, loss))
