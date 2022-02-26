# -*- coding:utf-8 -*-
# @Time : 2022/2/24 11:16 下午
# @Author : huichuan LI
# @File : nais.py
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


class NAIS(tf.keras.Model):
    """NAIS is an attention network, which is capable of distinguishing which historical items
       in a user profile are more important for a prediction. We just implement the model following
       the original author with a pointwise training mode.
       Note:
           instead of forming a minibatch as all training instances of a randomly sampled user which is
           mentioned in the original paper, we still train the model by a randomly sampled interactions.
       """

    def __init__(self, config, dataset):
        super().__init__()
        # load parameters info

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = np.max(dataset.rating[self.USER_ID]) + 1
        self.n_items = np.max(dataset.rating[self.ITEM_ID]) + 1

        self.embedding_size = config['embedding_size']
        self.weight_size = config['weight_size']
        self.algorithm = config['algorithm']
        self.reg_weights = config['reg_weights']
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.item_src_embedding = Embedding(self.n_items, self.embedding_size, mask_zero=0)
        self.item_dst_embedding = Embedding(self.n_items, self.embedding_size, mask_zero=0)
        if self.algorithm == 'concat':
            self.mlp_layers = tf.keras.layers.Dense(self.weight_size)
        elif self.algorithm == 'prod':
            self.mlp_layers = tf.keras.layers.Dense(self.weight_size)
        else:
            raise ValueError("NAIS just support attention type in ['concat', 'prod'] but get {}".format(self.algorithm))
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.output_dense = tf.keras.layers.Dense(1)
        self.opt = tf.keras.optimizers.Adam(0.01)

    def get_history_info(self, dataset):
        """get the user history interaction information
        Args:
            dataset (DataSet): train dataset
        Returns:
            tuple: (history_item_matrix, history_lens, mask_mat)
        """
        history_item_matrix, _, history_lens = dataset.history_item_matrix()
        arange_tensor = np.arange(history_item_matrix.shape[1])
        mask_mat = (arange_tensor < history_lens)
        return history_item_matrix, history_lens, mask_mat

    def call(self, X, training):
        """forward the model by interaction
        """
        user_inter = X[:, :-4]
        item = X[:, -3]
        item_num = X[:, -4]
        user_history = self.item_src_embedding(user_inter)  # batch_size x max_len x embedding_size
        print(user_history)
        target = self.item_dst_embedding(item)  # batch_size x embedding_size
        print(target)
        similarity = tf.reduce_sum(tf.linalg.matmul(user_history, target), axis=2)  # batch_size x max_len
        print(similarity)
        target = tf.expand_dims(target, axis=1)
        if self.algorithm == 'prod':
            mlp_input = tf.math.multiply(user_history, target)  # batch_size x max_len x embedding_size
        else:
            mlp_input = tf.concat(
                [user_history,
                 tf.broadcast_to(target, [target.shape[0], user_history.shape[1], target.shape[2]])],
                axis=2)  # batch_size x max_len x embedding_size*2
        mlp_output = self.mlp_layers(mlp_input)  # batch_size x max_len x weight_size
        logits = self.output_dense(mlp_output)
        scores = tf.expand_dims(self.mask_softmax(similarity, logits, item_num), axis=1)
        return scores

    def mask_softmax(self, similarity, logits, item_num):
        """softmax the unmasked user history items and get the final output
        Args:
            similarity (torch.Tensor): the similarity between the history items and target items
            logits (torch.Tensor): the initial weights of the history items
            item_num (torch.Tensor): user history interaction lengths
            bias (torch.Tensor): bias
            batch_mask_mat (torch.Tensor): the mask of user history interactions
        Returns:
            torch.Tensor: final output
        """

        exp_logits = tf.squeeze(tf.exp(logits), axis=2)  # batch_size x max_len
        # exp_logits = batch_mask_mat * exp_logits  # batch_size x max_len
        exp_sum = tf.math.reduce_sum(exp_logits, axis=1, keepdims=True)
        exp_sum = tf.math.pow(exp_sum, self.beta)
        weights = tf.math.divide(exp_logits, exp_sum)
        coeff = tf.cast(tf.math.pow(item_num, -self.alpha), tf.float32)
        output = tf.math.sigmoid(tf.multiply(coeff, tf.math.reduce_sum(weights * similarity, axis=1)))
        return output


if __name__ == "__main__":
    # 读取数据
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv",
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": "",
              "embedding_size": 10, 'weight_size': 64, 'algorithm': "concat", 'beta': 0.5,
              "alpha": 0, "reg_weights": [0, 0, 0], "dropout_prob": 0.8}

    dataset = Dataset(config=config)
    neuMF = NAIS(config, dataset=dataset)
    # print(history.summary())

    neuMF.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    data = dataset.rating

    history_item_matrix, _, history_lens = dataset.history_item_matrix()
    arange_tensor = np.arange(history_item_matrix.shape[1])
    # mask_mat = (arange_tensor < history_lens)
    print("history_matrix")
    print(history_item_matrix)
    print("length")
    print(history_lens)
    data["label"] = 0
    # print(data) g b
    data.label[data.rating > 4] = 1
    # print(data)
    data = data.iloc[:100000, :]
    data = data.to_numpy()
    data = np.column_stack((history_item_matrix[data[:, 0]], data))
    data = np.column_stack((history_lens[data[:, 0]], data))

    # print(data.shape)
    # user_inter = history_item_matrix[data[:10, 0]]
    # item_num = history_lens[data[:10, 0]]
    # print(user_inter.shape)
    # print(item_num)
    # # batch_mask_mat = mask_mat[data[:,0]]
    #

    neuMF.fit(data, data[:, -1], batch_size=2048, epochs=5)
