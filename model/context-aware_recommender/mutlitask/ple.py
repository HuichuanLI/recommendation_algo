# -*- coding:utf-8 -*-
# @Time : 2022/3/6 7:29 下午
# @Author : huichuan LI
# @File : ple.py
"""
Author:
    lihuichuan
Reference:
    [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//Fourteenth ACM Conference on Recommender Systems. 2020.(https://dl.acm.org/doi/10.1145/3383313.3412236)
"""

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform, TruncatedNormal)
from tensorflow.python.keras import backend as K
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
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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


def PLE(dnn_feature_columns, shared_expert_num=1, specific_expert_num=1, num_levels=2,
        expert_dnn_hidden_units=(256,), tower_dnn_hidden_units=(64,), gate_dnn_hidden_units=(),
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
        task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    num_tasks = len(task_names)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns))

    input_layer_dict = build_input_layers(dnn_feature_columns)
    input_layers = list(input_layer_dict.values())
    # 获取dense
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])

    dnn_dense_input = Concatenate(axis=1)(dnn_dense_input)

    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(sparse_feature_columns, input_layer_dict)

    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
                                                   flatten=True)

    # dnn_sparse_embed_input = [tf.expand_dims(elem, axis=1) for elem in dnn_sparse_embed_input]
    emb_input = Concatenate(axis=1)(dnn_sparse_embed_input)
    dnn_input = Concatenate(axis=1)([emb_input, dnn_dense_input])

    # single Extraction Layer
    def cgc_net(inputs, level_name, is_last=False):
        # inputs: [task1, task2, ... taskn, shared task]
        specific_expert_outputs = []
        # build task-specific expert layer
        for i in range(num_tasks):
            for j in range(specific_expert_num):
                expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                     seed=seed,
                                     name=level_name + 'task_' + task_names[i] + '_expert_specific_' + str(j))(
                    inputs[i])
                specific_expert_outputs.append(expert_network)

        # build task-shared expert layer
        shared_expert_outputs = []
        for k in range(shared_expert_num):
            expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                 seed=seed,
                                 name=level_name + 'expert_shared_' + str(k))(inputs[-1])
            shared_expert_outputs.append(expert_network)

        # task_specific gate (count = num_tasks)
        cgc_outs = []
        for i in range(num_tasks):
            # concat task-specific expert and task-shared expert
            cur_expert_num = specific_expert_num + shared_expert_num
            # task_specific + task_shared
            cur_experts = specific_expert_outputs[
                          i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs

            expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_specific_' + task_names[i])(
                inputs[i])  # gate[i] for task input[i]
            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_specific_' + task_names[i])(gate_input)
            gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis=1, keepdims=False),
                                                     name=level_name + 'gate_mul_expert_specific_' + task_names[i])(
                [expert_concat, gate_out])
            cgc_outs.append(gate_mul_expert)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = num_tasks * specific_expert_num + shared_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

            expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_shared')(inputs[-1])  # gate for shared task input

            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_shared')(gate_input)
            gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis=1, keepdims=False),
                                                     name=level_name + 'gate_mul_expert_shared')(
                [expert_concat, gate_out])

            cgc_outs.append(gate_mul_expert)
        return cgc_outs

        # build Progressive Layered Extraction

    ple_inputs = [dnn_input] * (num_tasks + 1)  # [task1, task2, ... taskn, shared task]
    ple_outputs = []
    for i in range(num_levels):
        if i == num_levels - 1:  # the last level
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=True)
        else:
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=False)
            ple_inputs = ple_outputs

    task_outs = []
    for task_type, task_name, ple_out in zip(task_types, task_names, ple_outputs):
        # build tower layer
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(ple_out)
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = tf.math.sigmoid(logit)
        task_outs.append(output)

    model = tf.keras.models.Model(inputs=input_layers, outputs=task_outs)

    return model


if __name__ == "__main__":
    # 读取数据
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    samples_data = pd.read_csv("../data/example.txt", sep=",", header=None, names=column_names)
    print(samples_data)
    samples_data['label_income'] = samples_data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    samples_data['label_marital'] = samples_data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    samples_data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)

    columns = samples_data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if
                      col not in sparse_features and col not in ['label_income', 'label_marital']]

    samples_data[sparse_features] = samples_data[sparse_features].fillna('-1', )
    samples_data[dense_features] = samples_data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    samples_data[dense_features] = mms.fit_transform(samples_data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        samples_data[feat] = lbe.fit_transform(samples_data[feat])

    fixlen_feature_columns = [SparseFeat(feat, samples_data[feat].max() + 1, embedding_dim=4) for feat in
                              sparse_features] \
                             + [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = sparse_features + dense_features

    # 3.generate input data for model

    train, test = train_test_split(samples_data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = PLE(fixlen_feature_columns, task_types=['binary', 'binary'],
                task_names=['label_income', 'label_marital'])

    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )
    print(model.summary())

    history = model.fit(train_model_input, [train['label_income'].values, train['label_marital'].values],
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
