# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 15:12
# @Author  : Li Huichuan
# @File    : AFM.py
# @Software: PyCharm
# -*- coding:utf-8 -*-
# @Time : 2022/2/28 11:19 下午
# @Author : huichuan LI
# @File : pnn.py
# @Software: PyCharm

"""
Reference:
    [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.(https://arxiv.org/pdf/1611.00144.pdf)
"""
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Layer
from utils import OutterProductLayer, InnerProductLayer
import pandas as pd
import numpy as np
from collections import namedtuple
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform, TruncatedNormal)
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2

SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])

from utils import DNN
import itertools


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


class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


class AFMLayer(Layer):
    """Attentonal Factorization Machine models pairwise (order-2) feature
    interactions without linear term and bias.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      Arguments
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.
        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.
        - **dropout_rate** : float between in [0,1). Fraction of the attention net output units to dropout.
        - **seed** : A Python integer to use as random seed.
      References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(self, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            # input_shape = input_shape[0]
            # if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')

        shape_set = set()
        reduced_input_shape = [shape.as_list() for shape in input_shape]
        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_input_shape[i]))

        if len(shape_set) > 1:
            raise ValueError('A `AttentionalFM` layer requires '
                             'inputs with same shapes '
                             'Got different shapes: %s' % (shape_set))

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A `AttentionalFM` layer requires '
                             'inputs of a list with same shape tensor like\
                             (None, 1, embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))

        embedding_size = int(input_shape[0][-1])

        self.attention_W = self.add_weight(shape=(embedding_size,
                                                  self.attention_factor), initializer=glorot_normal(seed=self.seed),
                                           regularizer=l2(self.l2_reg_w), name="attention_W")
        self.attention_b = self.add_weight(
            shape=(self.attention_factor,), initializer=Zeros(), name="attention_b")
        self.projection_h = self.add_weight(shape=(self.attention_factor, 1),
                                            initializer=glorot_normal(seed=self.seed), name="projection_h")
        self.projection_p = self.add_weight(shape=(
            embedding_size, 1), initializer=glorot_normal(seed=self.seed), name="projection_p")
        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate, seed=self.seed)

        self.tensordot = tf.keras.layers.Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

    def compute_output_shape(self, input_shape):

        if not isinstance(input_shape, list):
            raise ValueError('A `AFMLayer` layer should be called '
                             'on a list of inputs.')
        return (None, 1)

    def get_config(self, ):
        config = {'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(AFMLayer, self).get_config()
        base_config.update(config)
        return base_config

    def call(self, inputs, training=None, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = tf.concat(row, axis=1)
        q = tf.concat(col, axis=1)
        inner_product = p * q

        print(inner_product)
        bi_interaction = inner_product
        attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
            bi_interaction, self.attention_W, axes=(-1, 0)), self.attention_b))
        print(attention_temp)
        #  Dense(self.attention_factor,'relu',kernel_regularizer=l2(self.l2_reg_w))(bi_interaction)
        self.normalized_att_score = tf.nn.softmax(tf.tensordot(
            attention_temp, self.projection_h, axes=(-1, 0)), axis=1)
        attention_output = tf.reduce_sum(
            self.normalized_att_score * bi_interaction, axis=1)

        attention_output = self.dropout(attention_output, training=training)  # training

        afm_out = self.tensordot([attention_output, self.projection_p])
        return afm_out


def AFM(feature_columns, use_attention=True,
        attention_factor=8,
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0, seed=1024,
        task='binary'):
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

    if use_attention:
        fm_logit = tf.reduce_sum(AFMLayer(attention_factor, l2_reg_att, afm_dropout,
                                          seed)(dnn_sparse_embed_input))
    else:
        fm_logit = tf.reduce_sum(FM()(dnn_sparse_embed_input))

    input = Concatenate(axis=1)([emb_input, dnn_dense_input])
    dnn_input = dense_liner(input)

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_input)

    output = tf.math.sigmoid(dnn_logit + fm_logit)
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

    pnn = AFM(feature_columns)
    #
    pnn.compile('adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                         tf.keras.metrics.AUC()])
    pnn.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, )
    #
    print(pnn.summary())
