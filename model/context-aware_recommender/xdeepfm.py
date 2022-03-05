# -*- coding:utf-8 -*-
# @Time : 2022/2/28 11:18 下午
# @Author : huichuan LI
# @File : xdeepfm.py
"""
Author:
    lihuuchuan
Reference:
    [1] Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.(https://arxiv.org/pdf/1803.05170.pdf)
"""
from tensorflow.python.keras.initializers import RandomNormal

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Layer
from utils import OutterProductLayer, InnerProductLayer
import pandas as pd
import numpy as np
from collections import namedtuple
from tensorflow.python.keras.layers import (Dense, Embedding, Lambda,
                                            multiply)
from tensorflow.python.keras.regularizers import l2

SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])
import itertools
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


def feature_embedding(fc_i, fc_j, embedding_dict, input_feature):
    fc_i_embedding = embedding_dict[fc_i.name][fc_j.name](input_feature)
    return fc_i_embedding


def ONN(feature_columns, dnn_hidden_units=(256, 128, 64, 1),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, dnn_dropout=0,
        seed=1024, use_bn=True, reduce_sum=False, task='binary', ):
    input_layer_dict = build_input_layers(feature_columns)

    input_layers = list(input_layer_dict.values())

    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))

    sparse_embedding = {fc_j.name: {fc_i.name: Embedding(fc_j.vocabulary_size, fc_j.embedding_dim,
                                                         embeddings_initializer=RandomNormal(
                                                             mean=0.0, stddev=0.0001, seed=seed),
                                                         embeddings_regularizer=l2(
                                                             l2_reg_embedding),
                                                         mask_zero=isinstance(fc_j,
                                                                              VarLenSparseFeat),
                                                         name='sparse_emb_' + str(
                                                             fc_j.name) + '_' + fc_i.name)
                                    for fc_i in
                                    sparse_feature_columns} for fc_j in
                        sparse_feature_columns}
    embed_list = []
    for fc_i, fc_j in itertools.combinations(sparse_feature_columns, 2):
        i_input = input_layer_dict[fc_i.name]
        # if fc_i.use_hash:
        #     i_input = Hash(fc_i.vocabulary_size)(i_input)
        j_input = input_layer_dict[fc_j.name]
        # if fc_j.use_hash:
        #     j_input = Hash(fc_j.vocabulary_size)(j_input)

        fc_i_embedding = feature_embedding(fc_i, fc_j, sparse_embedding, i_input)
        fc_j_embedding = feature_embedding(fc_j, fc_i, sparse_embedding, j_input)

        element_wise_prod = multiply([fc_i_embedding, fc_j_embedding])
        if reduce_sum:
            element_wise_prod = Lambda(lambda element_wise_prod: K.sum(
                element_wise_prod, axis=-1))(element_wise_prod)
        embed_list.append(element_wise_prod)
    # 获取dense
    print(embed_list)
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])

    dnn_dense_input = Concatenate(axis=1)(dnn_dense_input)

    # 构建embedding字典
    # embedding_layer_dict = build_embedding_layers(feature_columns, input_layer_dict)

    # dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
    # flatten = True)

    # emb_input = Concatenate(axis=1)(dnn_sparse_embed_input)

    # input = Concatenate(axis=1)([emb_input, dnn_dense_input])
    ffm_out = tf.keras.layers.Flatten()(Concatenate(axis=1)(embed_list))
    print("ffm_out")
    print(ffm_out)
    if use_bn:
        ffm_out = tf.keras.layers.BatchNormalization()(ffm_out)
    dnn_out = Concatenate(axis=1)([ffm_out, dnn_dense_input])
    dnn_out = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout)(dnn_out)
    # print("dnn_output")
    # print(dnn_out)
    # dnn_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

    output = tf.math.sigmoid(dnn_out)
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

    onn = ONN(feature_columns)
    #
    onn.compile('adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.AUC()])
    onn.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, )
    #
    print(onn.summary())
