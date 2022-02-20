# -*- coding:utf-8 -*-
# @Time : 2021/9/20 1:51 下午
# @Author : huichuan LI
# @File : textlstm.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import *


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
output = TextLSTM(128,128, 0.8, 2)(embeddings)
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
