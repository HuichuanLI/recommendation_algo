# -*- coding:utf-8 -*-
# @Time : 2021/9/20 1:51 下午
# @Author : huichuan LI
# @File : textlstm.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import *


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention(step_dim))
        """
        self.supports_masking = True

        self.bias = bias
        self.step_dim = step_dim
        self.init = tf.keras.initializers.get('glorot_uniform')
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format("attention"),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format("attention"),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = tf.reshape(
            tf.keras.layers.Dot(axes=1)([tf.reshape(x, (-1, features_dim)), tf.reshape(self.W, (1, features_dim))]),
            (-1, step_dim))
        if self.bias:
            eij += self.b

        eij = tf.math.tanh(eij)

        a = tf.math.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= tf.cast(mask, tf.float64)

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= tf.cast(tf.math.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)
        a = tf.expand_dims(a, axis=-1)
        weighted_input = tf.math.multiply(x, a)
        return tf.math.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class TextLSTM(Layer):
    """textcnn实现
    """

    def __init__(self, hidden_size, emb_dim, dropout, class_num, **kwargs):
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.class_num = class_num

        super(TextLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TextLSTM, self).build(input_shape)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.hidden_size, return_sequences=True
        ))
        self.dropout = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs):
        x = inputs
        x = self.lstm(x)

        x = self.dropout(x)

        return x


from keras.preprocessing import sequence
from keras.datasets import imdb
import tensorflow as tf

max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=max_features, maxlen=maxlen
)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

S_inputs = tf.keras.Input(shape=(maxlen,), dtype='int32')
embeddings = tf.keras.layers.Embedding(max_features, 128)(S_inputs)
# embeddings = SinCosPositionEmbedding(128)(embeddings) # 增加Position_Embedding能轻微提高准确率
output = TextLSTM(128, 128, 0.8, 2)(embeddings)
output = Attention(maxlen)(output)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(output)
model = tf.keras.Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
