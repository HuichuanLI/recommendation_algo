# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 15:12
# @Author  : Li Huichuan
# @File    : AFM.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Layer


class AttLayer(Layer):
    """Calculate the attention signal(weight) according the input tensor.
    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].
    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = Dense(out_features=att_dim, bias=False)
        self.h = tf.Variable(tf.random.normal(att_dim), trainable=True)

    def forward(self, infeatures):
        att_signal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_signal = tf.nn.relu(att_signal)  # [batch_size, M, att_dim]

        att_signal = tf.matmul(att_signal, self.h)  # [batch_size, M, att_dim]
        att_signal = tf.reduce_mean(att_signal, axis=2)  # [batch_size, M]
        att_signal = tf.nn.softmax(att_signal, axis=1)  # [batch_size, M]

        return att_signal


class AFM(Model):
    """ AFM is a attention based FM model that predict the final score with the attention of input feature.
    """

    def __init__(self, config, dataset):
        super(AFM, self).__init__(config, dataset)

        # load parameters info
        self.attention_size = config['attention_size']
        self.dropout_prob = config['dropout_prob']
        self.reg_weight = config['reg_weight']
        self.num_pair = self.num_feature_field * (self.num_feature_field - 1) / 2

        # define layers and loss
        self.attlayer = AttLayer(self.embedding_size, self.attention_size)
        self.p = tf.Variable(tf.random.normal(self.embedding_size), trainable=True)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_prob)
        self.sigmoid = tf.nn.sigmoid()

    def build_cross(self, feat_emb):
        """ Build the cross feature columns of feature columns
        Args:
            feat_emb (torch.FloatTensor): input feature embedding tensor. shape of [batch_size, field_size, embed_dim].
        Returns:
            tuple:
                - torch.FloatTensor: Left part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
                - torch.FloatTensor: Right part of the cross feature. shape of [batch_size, num_pairs, emb_dim].
        """
        # num_pairs = num_feature_field * (num_feature_field-1) / 2
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]  # [batch_size, num_pairs, emb_dim]
        q = feat_emb[:, col]  # [batch_size, num_pairs, emb_dim]
        return p, q

    def afm_layer(self, infeature):
        """ Get the attention-based feature interaction score
        Args:
            infeature (torch.FloatTensor): input feature embedding tensor. shape of [batch_size, field_size, embed_dim].
        Returns:
            torch.FloatTensor: Result of score. shape of [batch_size, 1].
        """
        p, q = self.build_cross(infeature)
        pair_wise_inter = tf.matmul(p, q)  # [batch_size, num_pairs, emb_dim]

        # [batch_size, num_pairs, 1]
        att_signal = self.attlayer(pair_wise_inter).unsqueeze(dim=2)

        att_inter = tf.matmul(att_signal, pair_wise_inter)  # [batch_size, num_pairs, emb_dim]
        att_pooling = tf.reduce_sum(att_inter, axis=1)  # [batch_size, emb_dim]
        att_pooling = self.dropout_layer(att_pooling)  # [batch_size, emb_dim]

        att_pooling = tf.matmul(att_pooling, self.p)  # [batch_size, emb_dim]
        att_pooling = tf.reduce_sum(att_pooling, axis=1, keepdim=True)  # [batch_size, 1]

        return att_pooling

    def forward(self, interaction):
        afm_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]

        output = self.sigmoid(self.first_order_linear(interaction) + self.afm_layer(afm_all_embeddings))
        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        l2_loss = self.reg_weight * tf.norm(self.attlayer.w.weight, ord=2)
        return self.loss(output, label) + l2_loss

    def predict(self, interaction):
        return self.forward(interaction)
