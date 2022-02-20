# -*- coding:utf-8 -*-
# @Time : 2022/2/20 8:59 下午
# @Author : huichuan LI
# @File : neumf.py
# @Software: PyCharm
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


class NeuMF(tf.keras.Model):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """

    def __init__(self, config, dataset):
        super().__init__()
        # load parameters info

        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.use_pretrain = config['use_pretrain']
        if self.use_pretrain:
            self.mf_pretrain_path = config['mf_pretrain_path']
            self.mlp_pretrain_path = config['mlp_pretrain_path']

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = np.max(dataset.rating[self.USER_ID]) + 1
        self.n_items = np.max(dataset.rating[self.ITEM_ID]) + 1
        print(self.n_users)
        print(self.n_items)
        # define layers and loss
        self.user_mf_embedding = Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = Embedding(self.n_items, self.mlp_embedding_size)
        self.emb_dropout = tf.keras.layers.Dropout(self.dropout_prob)
        self.mlp_layers = tf.keras.layers.Dense(self.mlp_hidden_size)
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = tf.keras.layers.Dense(1)
        elif self.mf_train:
            self.predict_layer = tf.keras.layers.Dense(1)
        elif self.mlp_train:
            self.predict_layer = tf.keras.layers.Dense(1)

    def call(self, X, training):
        user_id = X[:, 0]
        item_id = X[:, 1]

        user_mf_e = self.user_mf_embedding(user_id)
        item_mf_e = self.item_mf_embedding(item_id)
        user_mlp_e = self.user_mlp_embedding(user_id)
        item_mlp_e = self.item_mlp_embedding(item_id)
        if self.mf_train:
            mf_output = tf.math.multiply(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(tf.concat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = tf.keras.activations.sigmoid(self.predict_layer(tf.concat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = tf.keras.activations.sigmoid(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = tf.keras.activations.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return output

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
              "mf_embedding_size": 10, 'mlp_embedding_size': 10, 'mlp_hidden_size': 32, 'use_pretrain': False,
              "mf_train": True, "mlp_train": True, "dropout_prob": 0.8}

    dataset = Dataset(config=config)
    neuMF = NeuMF(config, dataset=dataset)
    # print(history.summary())

    neuMF.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    data = dataset.rating
    data["label"] = 0
    print(data)
    data.label[data.rating > 4] = 1
    print(data)
    data = data.to_numpy()
    print(data)
    neuMF.fit(data[:, :2], data[:, 3], batch_size=1000, epochs=5)
