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
        self.RATING = config['LABEL_FIELD']

        # The dimensions of the last layer of users and items must be the same
        assert self.user_hidden_size_list == self.item_hidden_size_list
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = np.max(dataset.rating[self.USER_ID]) + 1
        self.n_items = np.max(dataset.rating[self.ITEM_ID]) + 1

        self.user_linear = tf.keras.layers.Dense(self.user_embedding_size)
        self.item_linear = tf.keras.layers.Dense(self.item_embedding_size)

        self.user_fc_layers = tf.keras.layers.Dense(self.user_hidden_size_list)
        self.item_fc_layers = tf.keras.layers.Dense(self.item_hidden_size_list)

        self.i_embedding = None

        # generate intermediate data
        self.history_user_id, self.history_user_value, _ = dataset.history_user_matrix(value_field=self.RATING)
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix(value_field=self.RATING)
        self.interaction_matrix = dataset.inter_matrix(form='csr', value_field=self.RATING).astype(np.float32)
        self.max_rating = self.history_user_value.max()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.opt = tf.keras.optimizers.Adam(0.01)

    def get_user_embedding(self, user):
        r"""Get a batch of user's embedding with the user's id and history interaction matrix.
        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]
        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        # col_indices = self.history_item_id[user].flatten()
        # row_indices = np.arange(len(user))
        # row_indices = np.repeat(row_indices, self.history_item_id.shape[1], axis=0)
        # matrix_01 = np.repeat(np.zeros(1), [len(user), self.n_items])
        # np.put(matrix_01, [row_indices, col_indices], self.history_item_value[user].flatten())
        matrix_01 = self.history_item_value[user]
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
        i = np.array([row, col])
        data = np.array(interaction_matrix.data)
        # item_matrix = torch.sparse.FloatTensor(i, data, torch.Size(interaction_matrix.shape)).to(self.device). \
        #     transpose(0, 1)
        item_matrix = data[i]
        item = self.item_fc_layers(item_matrix)
        return item

    def call(self, user_id, item_id):

        user = self.get_user_embedding(user_id)

        # Following lines construct tensor of shape [B,n_users] using the tensor of shape [B,H]
        # col_indices = self.history_user_id[user_id]
        # row_indices = np.repeat(np.arange(item_id.shape[0]), self.history_user_id.shape[1], axis=0)
        # matrix_01 = np.zeros ([len(item_id), self.n_users])
        # np.put(matrix_01, [row_indices, col_indices], self.history_user_value[item_id])
        matrix_01 = self.history_user_value[item_id]
        item = self.item_linear(matrix_01)
        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item)

        # cosine distance is replaced by dot product according the result of our experiments.
        vector = tf.math.reduce_sum(tf.multiply(user, item), axis=1)
        vector = tf.math.sigmoid(vector)
        return vector

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.call([user, item])

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        pass

    def calculate_loss(self, X, training=True):
        with tf.GradientTape() as tape:
            user = X["user_id"]
            item = X["item_id"]
            label = X["label"]
            output = self.call(user, item)

            label = label / self.max_rating  # normalize the label to calculate BCE loss.
            cur_loss = self.bce(output, label)
            grads = tape.gradient(cur_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return cur_loss.numpy()


if __name__ == "__main__":
    # 读取数据
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv", "k": 10,
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": "",
              "user_hidden_size_list": 100, 'item_hidden_size_list': 100, "dropout_prob": 0.8, "user_embedding_size": 10,
              "item_embedding_size": 10, }

    dataset = Dataset(config=config)
    dmf = DMF(config, dataset=dataset)
    # print(history.summary())

    data = dataset.rating
    data["label"] = 0
    print(data)
    data.label[data.rating > 4] = 1
    print(data)
    data = data.to_numpy()
    print(data)
    for t in range(10):
        for step in range(0, len(data), 10000):
            X = {"user_id": np.array(data[step:step + 1000,0]), \
                 "item_id": np.array(data[step: step + 1000,1]), \
                 "label": np.array(data[step: step + 1000,2])}

            loss = dmf.calculate_loss(X)
            if step % 100 == 0:
                print("epoch:{},step: {} | loss: {}".format(t, step, loss))
