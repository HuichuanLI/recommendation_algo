# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 15:33
# @Author  : Li Huichuan
# @File    : FM.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Layer

import pandas as pd
import numpy as np
from collections import namedtuple

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


class BaseFactorizationMachine(Layer):
    r"""Calculate FM result over the embeddings
    Args:
        reduce_sum: bool, whether to sum the result, default is True.
    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.
    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def call(self, input_x):
        square_of_sum = tf.reduce_sum(input_x, axis=1) ** 2
        sum_of_square = tf.reduce_sum(input_x ** 2, axis=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = tf.reduce_sum(output, axis=1, keepdims=True)
        output = 0.5 * output
        return output


def FM(feature_columns):
    input_layer_dict = build_input_layers(feature_columns)

    input_layers = list(input_layer_dict.values())

    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))

    # 获取dense
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])

    # 将所有的dense特征拼接
    dnn_dense_input = Concatenate(axis=1)(dnn_dense_input)
    dense_liner = Dense(1)
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(feature_columns, input_layer_dict)

    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
                                                   flatten=True)
    emb_input = Concatenate(axis=1)(dnn_sparse_embed_input)
    emb_input_fm = tf.reshape(emb_input, shape=[-1, 5, 8])
    fm = BaseFactorizationMachine()
    output = tf.reduce_sum(tf.math.sigmoid(fm(emb_input_fm) + dense_liner(emb_input)), axis=1)
    model = Model(input_layers, output)
    return model


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

    fm = FM(feature_columns)
    #
    fm.compile('adam',
               loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.AUC()])
    fm.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, )
    #
    # print(history.summary())
