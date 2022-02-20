# -*- coding:utf-8 -*-
# @Time : 2021/9/26 10:58 下午
# @Author : huichuan LI
# @File : fpmc.py
# @Software: PyCharm
r"""
FPMC
################################################
Reference:
    Steffen Rendle et al. "Factorizing Personalized Markov Chains for Next-Basket Recommendation." in WWW 2010.
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from collections import namedtuple
import pandas as pd
import numpy as np


def build_input_layers(feature_columns):
    input_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_layer_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            input_layer_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            input_layer_dict[fc.name] = Input(shape=(fc.maxlen,), name=fc.name)

    return input_layer_dict


class FPMC(Model):
    r"""The FPMC model is mainly used in the recommendation system to predict the possibility of
    unknown items arousing user interest, and to discharge the item recommendation list.
    Note:
        In order that the generation method we used is common to other sequential models,
        We set the size of the basket mentioned in the paper equal to 1.
        For comparison with other models, the loss function used is BPR.
    """

    def __init__(self, embedding_size, user_num, item_num):
        super(FPMC, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size

        # load dataset info
        self.n_users = user_num
        self.n_items = item_num

        # define layers and loss
        # user embedding matrix
        self.UI_emb = Embedding(self.n_users, self.embedding_size)
        # label embedding matrix
        self.IU_emb = Embedding(self.n_items, self.embedding_size)
        # last click item embedding matrix
        self.LI_emb = Embedding(self.n_items, self.embedding_size)
        # label embedding matrix
        self.IL_emb = Embedding(self.n_items, self.embedding_size)

    def call(self, feature_columns):
        user = feature_columns["user_id"]
        item_last_click = feature_columns["item_last_click"]
        next_item = feature_columns["next_item"]

        item_seq_emb = self.LI_emb(item_last_click)  # [b,1,emb]

        user_emb = self.UI_emb(user)

        iu_emb = self.IU_emb(next_item)

        il_emb = self.IL_emb(next_item)

        # This is the core part of the FPMC model,can be expressed by a combination of a MF and a FMC model
        #  MF
        mf = tf.matmul(user_emb, tf.transpose(iu_emb, perm=(0, 2, 1)))
        mf = tf.squeeze(mf, axis=1)  # [B,1]
        #  FMC
        fmc = tf.matmul(il_emb, tf.transpose(item_seq_emb, perm=(0, 2, 1)))
        fmc = tf.squeeze(fmc, axis=1)  # [B,1]

        score = mf + fmc
        score = tf.squeeze(score)
        output = tf.math.sigmoid(score)

        return output


if __name__ == "__main__":
    # 读取数据
    samples_data = pd.read_csv("./data/movie_sample.txt", sep="\t", header=None)
    print(samples_data.shape)
    samples_data.columns = ["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id",
                            "label"]

    # samples_data = shuffle(samples_data)

    X = samples_data[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
    y = samples_data["label"]

    X_train = {"user_id": np.array(X["user_id"]), \
               "item_last_click": np.array([[int(l.split(',')[-1])] for l in X["hist_movie_id"]]), \
               "next_item": np.array(X["movie_id"])}

    y_train = np.array(y)
    n_users = max(samples_data["user_id"]) + 1
    n_item = max(samples_data["movie_id"]) + 1

    print(n_users, n_item)

    # 使用具名元组定义特征标记
    SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
    DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
    VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])

    feature_columns = [SparseFeat('user_id', max(samples_data["user_id"]) + 1, embedding_dim=8),
                       SparseFeat('next_item', max(samples_data["movie_id"]) + 1, embedding_dim=8),
                       SparseFeat('item_last_click', max(samples_data["movie_id"]) + 1, embedding_dim=8)]

    feature_columns += [
        VarLenSparseFeat('hist_movie_id', vocabulary_size=max(samples_data["movie_id"]) + 1, embedding_dim=8,
                         maxlen=50)]
#
# 行为特征列表，表示的是基础特征
behavior_feature_list = ['movie_id']
# 行为序列特征
behavior_seq_feature_list = ['hist_movie_id']

# 构建Input层
input_layer_dict = build_input_layers(feature_columns)
print(input_layer_dict.values())

from tensorflow.keras import optimizers

history = FPMC(embedding_size=8, user_num=n_users, item_num=n_item)

#
history.compile('adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.AUC()])
history.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, )
print(history.summary())
