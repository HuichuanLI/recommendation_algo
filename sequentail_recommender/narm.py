# -*- coding:utf-8 -*-
# @Time : 2021/9/29 11:15 下午
# @Author : huichuan LI
# @File : narm.py
# @Software: PyCharm

r"""
GRU4Rec
################################################
Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.
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


class NARM(Model):
    r"""The FPMC model is mainly used in the recommendation system to predict the possibility of
    unknown items arousing user interest, and to discharge the item recommendation list.
    Note:
        In order that the generation method we used is common to other sequential models,
        We set the size of the basket mentioned in the paper equal to 1.
        For comparison with other models, the loss function used is BPR.
    """

    def __init__(self, embedding_size, hidden_size, num_layers, dropout_prob, n_items):
        super(NARM, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.n_items = n_items

        # define layers and loss
        # user embedding matrix

        self.emb_dropout = Dropout(self.dropout_prob)

        self.item_embedding = Embedding(self.n_items, self.embedding_size)
        self.gru = [tf.keras.layers.GRU(
            self.hidden_size, activation='tanh', recurrent_activation='sigmoid',
            use_bias=True, kernel_initializer='glorot_uniform', return_sequences=True,
            return_state=True
        ) for i in range(self.num_layers)]
        self.a_1 = Dense(self.hidden_size, use_bias=False)
        self.a_2 = Dense(self.hidden_size, use_bias=False)
        self.v_t = Dense(1, use_bias=False)
        self.b = Dense(self.embedding_size, use_bias=False)

    def call(self, feature_columns):
        item_seq = feature_columns["hist_movie_id"]
        next_item = feature_columns["next_item"]

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        state = None
        for i in range(self.num_layers):
            seqs, state = self.gru[i](item_seq_emb_dropout, initial_state=state)
        gru_output = seqs[:, -1, :]
        c_global = gru_output
        q1 = self.a_1(seqs)
        q2 = self.a_2(gru_output)
        # mask = item_seq.gt(0).unsqueeze(2).expand_as(seqs)
        q2_expand = tf.expand_dims(q2, axis=1)

        # alpha = self.v_t(mask * tf.math.sigmoid(q1 + q2_expand))
        alpha = self.v_t(tf.math.sigmoid(q1 + q2_expand))

        c_local = tf.reduce_sum(alpha * seqs, axis=1)
        c_t = tf.concat([c_local, c_global], axis=1)
        c_t = self.emb_dropout(c_t)
        seq_output = self.b(c_t)
        item_next = tf.squeeze(self.item_embedding(next_item), axis=1)
        logits = tf.reduce_sum(tf.multiply(seq_output, item_next), axis=1)
        output = tf.math.sigmoid(logits)

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
               "hist_movie_id": np.array([[int(e) for e in l.split(',')] for l in X["hist_movie_id"]]), \
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
    print(input_layer_dict.keys())

    from tensorflow.keras import optimizers

    history = NARM(embedding_size=8, hidden_size=10, num_layers=2, dropout_prob=0.8, n_items=n_item)

    #
    history.compile('adam',
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.BinaryAccuracy(),
                             tf.keras.metrics.AUC()])
    history.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, )
    print(history.summary())
