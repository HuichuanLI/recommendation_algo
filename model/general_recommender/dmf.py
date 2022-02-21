# -*- coding:utf-8 -*-
# @Time : 2022/2/21 11:47 下午
# @Author : huichuan LI
# @File : dmf.py
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


class DMF(tf.keras.Model):
    r"""DMF is an neural network enhanced matrix factorization model.
    The original interaction matrix of :math:`n_{users} \times n_{items}` is set as model input,
    we carefully design the data interface and use sparse tensor to train and test efficiently.
    We just implement the model following the original author with a pointwise training mode.

    """

    def __init__(self, config, dataset):
        super().__init__()
        # load parameters info

        self.user_embedding_size = config['user_embedding_size']
        self.item_embedding_size = config['item_embedding_size']
        self.user_hidden_size_list = config['user_hidden_size_list']
        self.item_hidden_size_list = config['item_hidden_size_list']
        # The dimensions of the last layer of users and items must be the same
        assert self.user_hidden_size_list[-1] == self.item_hidden_size_list[-1]
        self.inter_matrix_type = config['inter_matrix_type']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = np.max(dataset.rating[self.USER_ID]) + 1
        self.n_items = np.max(dataset.rating[self.ITEM_ID]) + 1

        self.user_linear = tf.keras.layers.Dense(self.user_embedding_size, bias=False)
        self.item_linear = tf.keras.layers.Dense(self.user_embedding_size, bias=False)

        self.user_fc_layers = tf.keras.layers.Dense(self.user_hidden_size_list)
        self.item_fc_layers = tf.keras.layers.Dense(self.item_hidden_size_list)

        self.i_embedding = None

        # generate intermediate data
        self.history_user_id, self.history_user_value, _ = dataset.history_user_matrix(value_field=self.RATING)
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix(value_field=self.RATING)
        self.interaction_matrix = dataset.inter_matrix(form='csr', value_field=self.RATING).astype(np.float32)
        self.max_rating = self.history_user_value.max()

    def call(self, X, training):
        user_id = X[:, 0]
        item_id = X[:, 1]

        user = self.get_user_embedding(user_id)

        # Following lines construct tensor of shape [B,n_users] using the tensor of shape [B,H]
        col_indices = self.history_user_id[item].flatten()
        row_indices = torch.arange(item.shape[0]).to(self.device). \
            repeat_interleave(self.history_user_id.shape[1], dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(item.shape[0], self.n_users)
        matrix_01.index_put_((row_indices, col_indices), self.history_user_value[item].flatten())
        item = self.item_linear(matrix_01)

        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item)

        # cosine distance is replaced by dot product according the result of our experiments.
        vector = torch.mul(user, item).sum(dim=1)
        vector = self.sigmoid(vector)

    def get_user_embedding(self, user):
        r"""Get a batch of user's embedding with the user's id and history interaction matrix.
        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]
        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(self.history_item_id.shape[1], dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        matrix_01.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        user = self.user_linear(matrix_01)

        return user

    def get_item_embedding(self):
        r"""Get all item's embedding with history interaction matrix.
        Considering the RAM of device, we use matrix multiply on sparse tensor for generalization.
        Returns:
            torch.FloatTensor: The embedding tensor of all item, shape: [n_items, embedding_size]
        """
        interaction_matrix = self.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(interaction_matrix.data)
        item_matrix = torch.sparse.FloatTensor(i, data, torch.Size(interaction_matrix.shape)).to(self.device). \
            transpose(0, 1)
        item = torch.sparse.mm(item_matrix, self.item_linear.weight.t())

        item = self.item_fc_layers(item)
        return item

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.call([user, item])

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
