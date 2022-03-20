# -*- coding:utf-8 -*-
# @Time : 2022/3/20 1:43 下午
# @Author : huichuan LI
# @File : metaheac.py
# @Software: PyCharm
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


def HeacModel(dnn_feature_columns, num_experts=3, critic_num=3, expert_dnn_hidden_units=(256, 128),
              tower_dnn_hidden_units=(64, 1),
              gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
              dnn_activation='relu',
              dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
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

    task_embedding = tf.Variable(initial_value=tf.random.normal(shape=(num_tasks, 8)), shape=(num_tasks, 8))
    expert = [DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                  name='expert_' + str(i)) for i in range(num_experts)]
    critic = [DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                  name='tower_' + str(i)) for i in range(critic_num)]

    expert_gate = tf.keras.layers.Dense(num_experts, activation='softmax')
    critic_gate = tf.keras.layers.Dense(critic_num, activation='softmax')

    tasks_output = []
    # build expert layer
    gate_input_emb = []
    for i in range(num_tasks):
        dim = tf.shape(dnn_input)[0]
        dnn_input_task = Concatenate(axis=1)(
            [dnn_input, tf.tile(tf.expand_dims(task_embedding[i], axis=0), [dim, 1])])
        gate_input_emb.append(dnn_input_task)

    expert_gate_value = [tf.expand_dims(expert_gate(gate_input_emb[i]), axis=1) for i in range(num_tasks)]
    print(expert_gate_value)
    fea = Concatenate(axis=1)([tf.expand_dims(expert[i](emb_input), axis=1) for i in range(num_experts)])
    print(fea)
    task_fea = [tf.squeeze(tf.linalg.matmul(expert_gate_value[i], fea), axis=1) for i in range(num_tasks)]
    print("task_fea")
    print(task_fea)
    critic_gate_value = [critic_gate(gate_input_emb[i]) for i in range(num_tasks)]
    print(critic_gate_value)

    task_outs = []
    for i in range(num_tasks):
        output = [tf.math.sigmoid(critic[j](task_fea[i])) for j in range(critic_num)]
        output = Concatenate(axis=1)(output)
        print("output")
        print(output)
        print("output+2")
        print(tf.math.reduce_mean(critic_gate_value[i] * output, axis=1))
        task_outs.append(tf.expand_dims(tf.math.reduce_mean(critic_gate_value[i] * output, axis=1), axis=1))

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
    model = HeacModel(fixlen_feature_columns, task_types=['binary', 'binary'],
                      task_names=['label_income', 'label_marital'])

    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )
    print(model.summary())

    history = model.fit(train_model_input, [train['label_income'].values, train['label_marital'].values],
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
