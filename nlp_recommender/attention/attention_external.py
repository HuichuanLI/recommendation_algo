# -*- coding:utf-8 -*-
# @Time : 2021/9/19 10:53 下午
# @Author : huichuan LI
# @File : attention_external.py
# @Software: PyCharm
import numpy as np

from tensorflow.keras.layers import *
import tensorflow as tf


class ExternalAttention(Layer):
    def __init__(self, model, s=64, **kwargs):
        self.model = model
        self.s = s
        super(ExternalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ExternalAttention, self).build(input_shape)
        self.mk = tf.keras.layers.Dense(self.s, use_bias=False)
        self.mv = tf.keras.layers.Dense(self.model, use_bias=False)

    def call(self, x):
        attn = self.mk(x)  # bs,n,S
        attn = tf.nn.softmax(attn)  # bs,n,S
        attn = attn / tf.reduce_sum(attn, axis=2, keepdims=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model
        return out


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

S_inputs = tf.keras.Input(shape=(None,), dtype='int32')
embeddings = tf.keras.layers.Embedding(max_features, 128)(S_inputs)
# embeddings = SinCosPositionEmbedding(128)(embeddings) # 增加Position_Embedding能轻微提高准确率
O_seq = ExternalAttention(512, 8)(embeddings)
O_seq = tf.keras.layers.GlobalAveragePooling1D()(O_seq)
O_seq = tf.keras.layers.Dropout(0.5)(O_seq)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(O_seq)

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
