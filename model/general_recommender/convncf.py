# -*- coding:utf-8 -*-
# @Time : 2022/2/20 10:23 下午
# @Author : huichuan LI
# @File : convncf.py
# @Software: PyCharm
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


def ConvNCFBPRLoss(pos_score, neg_score):
    distance = pos_score - neg_score
    loss = tf.sum(tf.math.log((1 + tf.math.exp(-distance))))
    return loss


class ConvNCF(tf.keras.Model):
    r"""ConvNCF is a a new neural network framework for collaborative filtering based on NCF.
    It uses an outer product operation above the embedding layer,
    which results in a semantic-rich interaction map that encodes pairwise correlations between embedding dimensions.
    We carefully design the data interface and use sparse tensor to train and test efficiently.
    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, config, dataset):
        super().__init__()
        # load parameters info

        self.embedding_size = config['embedding_size']
        self.cnn_channels = config['cnn_channels']
        self.cnn_kernels = config['cnn_kernels']
        self.cnn_strides = config['cnn_strides']
        self.dropout_prob = config['dropout_prob']
        # self.regs = config['reg_weights']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = np.max(dataset.rating[self.USER_ID]) + 1
        self.n_items = np.max(dataset.rating[self.ITEM_ID]) + 1
        # define layers and loss
        self.user_embedding = Embedding(self.n_users, self.embedding_size)
        self.item_embedding = Embedding(self.n_items, self.embedding_size)
        self.emb_dropout = tf.keras.layers.Dropout(self.dropout_prob)

        self.num_of_nets = len(self.cnn_channels) - 1
        self.cnn_modules = []

        for i in range(self.num_of_nets):
            self.cnn_modules.append(
                tf.keras.layers.Conv2D(self.cnn_channels[i], (self.cnn_kernels[i], self.cnn_kernels[i]),
                                       strides=(self.cnn_strides[i], self.cnn_strides[i]), activation='relu',
                                       padding="same")
            )
        self.predict_layers = tf.keras.layers.Dense(1)

    def call(self, X, training):
        user_id = X[:, 0]
        item_id = X[:, 1]

        user_e = self.user_embedding(user_id)
        item_e = self.item_embedding(item_id)

        interaction_map = tf.linalg.matmul(tf.expand_dims(user_e, axis=2), tf.expand_dims(item_e, axis=1))
        interaction_map = tf.expand_dims(interaction_map, axis=-1)
        print(interaction_map)
        for i in range(self.num_of_nets):
            interaction_map = self.cnn_modules[i](interaction_map)
            print(interaction_map)

        cnn_output = interaction_map
        cnn_output = tf.squeeze(cnn_output, axis=(1, 2))
        prediction = self.predict_layers(cnn_output)
        return prediction

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return tf.matmul(user_e, item_e, transpose_b=True)



if __name__ == "__main__":
    # 读取数据
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv", "k": 10,
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": "",
              "embedding_size": 10, "dropout_prob": 0.8, 'cnn_channels': [1, 128, 128, 64, 32],
              "cnn_kernels": [4, 4, 2, 2], 'cnn_strides': [4, 4, 2, 2]}

    dataset = Dataset(config=config)
    convncf = ConvNCF(config, dataset=dataset)

    convncf.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy'])

    data = dataset.rating
    data["label"] = 0
    # print(data)
    data.label[data.rating > 4] = 1
    # print(data)
    data = data.to_numpy()
    # print(data)
    convncf.fit(data[:, :2], data[:, 3], batch_size=1000, epochs=5)
