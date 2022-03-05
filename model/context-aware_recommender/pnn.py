# -*- coding:utf-8 -*-
# @Time : 2022/2/28 11:19 下午
# @Author : huichuan LI
# @File : pnn.py
# @Software: PyCharm

"""
Author:
    lihuuchuan
Reference:
    [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.(https://arxiv.org/pdf/1611.00144.pdf)
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Layer
from utils import OutterProductLayer, InnerProductLayer
import pandas as pd
import numpy as np
from collections import namedtuple

SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])

from utils import DNN


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


# 将所有的sparse特征embedding拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    embedding_list = []
    for fc in feature_columns:
        _input = input_layer_dict[fc.name]  # 获取输入层
        _embed = embedding_layer_dict[fc.name]  # B x 1 x dim  获取对应的embedding层
        embed = _embed(_input)  # B x dim  将input层输入到embedding层中

        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，否则不需要
        if flatten:
            embed = Flatten()(embed)

        embedding_list.append(embed)

    return embedding_list


# 构建embedding层
def build_embedding_layers(feature_columns, input_layer_dict):
    embedding_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='emb_' + fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size + 1, fc.embedding_dim, name='emb_' + fc.name,
                                                      mask_zero=True)

    return embedding_layer_dict


def PNN(dnn_feature_columns, dnn_hidden_units=(256, 128, 64), l2_reg_embedding=0.00001, l2_reg_dnn=0,
        seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False, kernel_type='mat',
        task='binary'):
    if kernel_type not in ['mat', 'vec', 'num']:
        raise ValueError("kernel_type must be mat,vec or num")
    input_layer_dict = build_input_layers(feature_columns)

    input_layers = list(input_layer_dict.values())

    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))

    # 获取dense
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])
    print(dnn_dense_input)
    # 将所有的dense特征拼接
    dnn_dense_input = Concatenate(axis=1)(dnn_dense_input)
    dense_liner = Dense(1)

    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(feature_columns, input_layer_dict)

    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
                                                   flatten=True)

    emb_input = Concatenate(axis=1)(dnn_sparse_embed_input)

    dnn_sparse_embed_input = [tf.expand_dims(elem, axis=1) for elem in dnn_sparse_embed_input]

    inner_product = tf.keras.layers.Flatten()(
        InnerProductLayer()(dnn_sparse_embed_input))
    outter_product = OutterProductLayer(kernel_type)(dnn_sparse_embed_input)

    print(inner_product)
    print(outter_product)

    input = Concatenate(axis=1)([emb_input, dnn_dense_input])
    dnn_input = dense_liner(input)

    # linear_signal = tf.keras.layers.Reshape(
    #     [sum(map(lambda x: int(x.shape[-1]), dnn_sparse_embed_input))])(concat_func(dnn_sparse_embed_input))

    if use_inner and use_outter:
        deep_input = tf.keras.layers.Concatenate()(
            [dnn_input, inner_product, outter_product])
    elif use_inner:
        deep_input = tf.keras.layers.Concatenate()(
            [dnn_input, inner_product])
    elif use_outter:
        deep_input = tf.keras.layers.Concatenate()(
            [dnn_input, outter_product])
    else:
        deep_input = dnn_input
    # emb_input = Concatenate(axis=1)(dnn_sparse_embed_input)
    #
    # input = Concatenate(axis=1)([emb_input, dnn_dense_input])

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(deep_input)

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    output = tf.math.sigmoid(dnn_logit)
    model = Model(input_layers, output)
    return model


if __name__ == "__main__":
    # 读取数据

    samples_data = pd.read_csv("data/movie_sample.txt", sep="\t", header=None)
    print(samples_data.shape)
    samples_data.columns = ["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id",
                            "label"]

    # samples_data = shuffle(samples_data)

    X = samples_data[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
    y = samples_data["label"]

    X_train = {"user_id": np.array(X["user_id"]), \
               "gender": np.array(X["gender"]), \
               "age": np.array(X["age"]), \
               "hist_len": np.array(X["hist_len"]), \
               "movie_id": np.array(X["movie_id"]), \
               "movie_type_id": np.array(X["movie_type_id"])}

    y_train = np.array(y)

    feature_columns = [SparseFeat('user_id', max(samples_data["user_id"]) + 1, embedding_dim=8),
                       SparseFeat('gender', max(samples_data["gender"]) + 1, embedding_dim=8),
                       SparseFeat('age', max(samples_data["age"]) + 1, embedding_dim=8),
                       SparseFeat('movie_id', max(samples_data["movie_id"]) + 1, embedding_dim=8),
                       SparseFeat('movie_type_id', max(samples_data["movie_type_id"]) + 1, embedding_dim=8),
                       DenseFeat('hist_len', 1)]

    print(X_train)
    n_users = max(samples_data["user_id"]) + 1
    n_item = max(samples_data["movie_id"]) + 1

    pnn = PNN(feature_columns)
    #
    pnn.compile('adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.AUC()])
    pnn.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, )
    #
    print(pnn.summary())
