# -*- coding:utf-8 -*-
# @Time : 2022/3/5 10:56 上午
# @Author : huichuan LI
# @File : mlr.py
# @Software: PyCharm
# -*- coding:utf-8 -*-
# @Time : 2022/2/28 11:19 下午
# @Author : huichuan LI
# @File : dcn.py
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import batch_dot
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform, TruncatedNormal)
import pandas as pd
import numpy as np
from collections import namedtuple
from utils import DNN
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Activation, dot

"""
Reference:
    [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
"""
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])


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


def get_region_score(features, region_number):
    region_logit = Concatenate(axis=1)([Dense(1)(features) for i in
                                        range(region_number)])
    return Activation('softmax')(region_logit)


def get_learner_score(features, region_number):
    region_logit = Concatenate(axis=1)([Dense(1)(features) for i in
                                        range(region_number)])
    return Activation('sigmoid')(region_logit)


def MLR(feature_columns, base_feature_columns=None, region_num=4,
        l2_reg_linear=1e-5, seed=1024, task='binary',
        bias_feature_columns=None):
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
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(feature_columns, input_layer_dict)

    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
                                                   flatten=True)
    emb_input = Concatenate(axis=1)(dnn_sparse_embed_input)

    input = Concatenate(axis=1)([emb_input, dnn_dense_input])

    region_score = get_region_score(input, region_num)
    learner_score = get_learner_score(input, region_num)

    final_logit = dot([region_score, learner_score], axes=-1)

    output = tf.math.sigmoid(tf.keras.layers.Dense(1)(input) + final_logit)
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

    mlr = MLR(feature_columns)
    #
    mlr.compile('adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.AUC()])
    mlr.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, )
    #
    print(mlr.summary())
