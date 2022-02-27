# -*- coding:utf-8 -*-
# @Time : 2022/2/27 4:54 下午
# @Author : huichuan LI
# @File : fism.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from collections import namedtuple
import pandas as pd
import numpy as np
from model.abstract_model import GeneralRecommender
from dataset import Dataset


class FISM(tf.keras.Model):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """

    def __init__(self, config, dataset):
        super().__init__()
        # load parameters info

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = np.max(dataset.rating[self.USER_ID]) + 1
        self.n_items = np.max(dataset.rating[self.ITEM_ID]) + 1
        print(self.n_users)
        print(self.n_items)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.reg_weights = config['reg_weights']
        self.alpha = config['alpha']

        # define layers and loss
        # construct source and destination item embedding matrix
        self.item_src_embedding = Embedding(self.n_items, self.embedding_size)
        self.item_dst_embedding = Embedding(self.n_items, self.embedding_size)

        self.user_bias = self.add_weight(name='w',
                                         shape=(self.n_users, 1),
                                         initializer='zero',
                                         trainable=True)
        self.item_bias = self.add_weight(name='bias',
                                         shape=(self.n_items, 1),
                                         initializer='zero',
                                         trainable=True)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, X, training):
        user_inter = X["user_inter"]
        item = X["item"]
        item_num = X["item_num"]
        user = X["user"]

        # user_inter = Input(shape=(10228,), name="user_inter")
        # item = Input(shape=(1,), name="item")
        # item_num = Input(shape=(1,), name="item_num")
        # user = Input(shape=(1,), name="user")
        user_history = self.item_src_embedding(user_inter)  # batch_size x max_len x embedding_size
        target = self.item_dst_embedding(item)  # batch_size x embedding_size
        user_bias = tf.gather(self.user_bias, tf.cast(user, tf.int32))  # batch_size x 1
        item_bias = tf.gather(self.item_bias, tf.cast(item, tf.int32))
        similarity = tf.squeeze(tf.matmul(user_history, target, transpose_b=True), axis=2)  # batch_size x max_len
        # similarity = batch_mask_mat * similarity
        coeff = tf.cast(tf.squeeze(tf.math.pow(item_num, -self.alpha), axis=1), tf.float32)
        scores = tf.expand_dims(tf.math.sigmoid(tf.multiply(coeff, tf.reduce_sum(similarity, axis=1))), axis=1)
        return scores

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward(user, item)
        return output

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        pass


if __name__ == "__main__":
    # 读取数据
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv", "k": 10,
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": "",
              "mf_embedding_size": 10, 'embedding_size': 10, 'reg_weights': 0.5, 'use_pretrain': False,
              "mf_train": True, "mlp_train": True, "dropout_prob": 0.8, "alpha": 0}

    dataset = Dataset(config=config)
    fism = FISM(config, dataset=dataset)
    # print(history.summary())

    fism.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])

    history_item_matrix, _, history_lens = dataset.history_item_matrix()
    arange_tensor = np.arange(history_item_matrix.shape[1])
    # mask_mat = (arange_tensor < history_lens)
    print("history_matrix")
    print(history_item_matrix)
    print("length")
    print(history_lens)
    data = dataset.rating

    data["label"] = 0
    # print(data) g b
    data.label[data.rating > 4] = 1
    # print(data)
    data = data.iloc[:10000, :]
    data = data.to_numpy()
    # data = np.column_stack((history_item_matrix[data[:, 0]], data))
    # data = np.column_stack((history_lens[data[:, 0]], data))
    print(data[:, -1])

    X_train = {"user_inter": np.array(history_item_matrix[data[:, 0]]), \
               "item": np.array(data[:, 1]), \
               "user": np.array(data[:, 0]), \
               "item_num": np.array(history_lens[data[:, 0]])}

    fism.fit(X_train, data[:, -1], batch_size=1000, epochs=5)
