# -*- coding:utf-8 -*-
# @Time : 2022/2/27 4:20 下午
# @Author : huichuan LI
# @File : multivae.py
# @Software: PyCharm
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


class Sampling(Layer):
    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        std = tf.math.exp(0.5 * z_log_var)

        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0, stddev=0.01)
        return z_mean + epsilon * std


class MultiVAE(tf.keras.Model):
    r"""MultiVAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.
    We implement the MultiVAE model with only user dataloader.
    """

    def __init__(self, config, dataset):
        super().__init__()
        # load parameters info
        self.update = 0
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = np.max(dataset.rating[self.USER_ID]) + 1
        self.n_items = np.max(dataset.rating[self.ITEM_ID]) + 1
        print(self.n_users)
        print(self.n_items)

        # load parameters info
        self.hidden_size = config["mlp_hidden_size"]
        self.lat_dim = config['latent_dimension']
        self.drop_out = config['dropout_prob']
        self.drop_out_layers = tf.keras.layers.Dropout(self.drop_out)
        self.anneal_cap = config['anneal_cap']
        self.total_anneal_steps = config["total_anneal_steps"]

        self.encode_layer_dims = self.hidden_size
        self.decode_layer_dims = self.hidden_size[::-1] + [10227]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)
        self.reparameterize = Sampling()
        self.opt = tf.keras.optimizers.Adam(0.01)

        # define layers and loss
        # construct source and destination item embedding matrix

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i in layer_dims:
            mlp_modules.append(tf.keras.layers.Dense(i, activation="relu"))
        return mlp_modules

    def reparameterize(self, mu, logvar, training):
        if training:
            std = tf.math.exp(0.5 * logvar)
            epsilon = tf.random.normal(std.shape, mean=0, stddev=0.01)
            return mu + epsilon * std
        else:
            return mu

    def calculate_loss(self, X):

        with tf.GradientTape() as tape:
            rating_matrix = X["rating_matrix"]

            self.update += 1
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            z, mu, logvar = self.call(rating_matrix)
            # print("z")
            # print(z)
            # print("mi")
            # print(mu)
            # print("logvar")
            # print(logvar)
            # print(tf.math.reduce_sum(1 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar), axis=1))
            # KL loss
            kl_loss = -0.5 * tf.math.reduce_mean(
                tf.math.reduce_sum(1 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar), axis=1)) * anneal
            # print("kl_loss")
            # print(kl_loss)
            # CE loss
            # print(-(tf.nn.log_softmax(z, 1)))
            ce_loss = tf.math.reduce_mean(tf.reduce_sum(-(tf.nn.log_softmax(z, 1) * rating_matrix), axis=1))
            # print("ce_loss")
            # print(ce_loss)
            cur_loss = ce_loss + kl_loss

            grads = tape.gradient(cur_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return cur_loss.numpy()

    def call(self, X):
        # rating_matrixs = Input(shape=(10228,), name="user_inter")
        rating_matrixs = X
        h = tf.clip_by_value(tf.keras.layers.Normalization(axis=1)(rating_matrixs), 1e-10, 1.0)

        h = self.drop_out_layers(h)

        for elem in self.encoder:
            h = elem(h)

        mu = h[:, :int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2):]
        z = self.reparameterize(mu, logvar)
        for elem in self.decoder:
            z = elem(z)

        return z, mu, logvar

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        pass


if __name__ == "__main__":
    # 读取数据
    print(tf.__version__)
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv", "k": 10,
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": "",
              "mlp_hidden_size": [128], 'embedding_size': 10, 'latent_dimension': 128, 'use_pretrain': False,
              "anneal_cap": 0.2, "total_anneal_steps": 200000, "dropout_prob": 0.8, "alpha": 0}

    dataset = Dataset(config=config)
    vae = MultiVAE(config, dataset=dataset)
    history_item_matrix, _, history_lens = dataset.history_item_matrix()
    print(history_item_matrix.shape)
    for t in range(10):
        for step in range(0, len(history_item_matrix), 10000):
            X = {"rating_matrix": history_item_matrix[step:step + 10000]}

            loss = vae.calculate_loss(X)
            if step % 100 == 0:
                print("epoch:{},step: {} | loss: {}".format(t, step, loss))
