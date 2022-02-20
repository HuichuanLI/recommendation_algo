# -*- coding:utf-8 -*-
# @Time : 2021/9/20 10:07 上午
# @Author : huichuan LI
# @File : textcnn.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.keras.layers import *


class TextCNN(Layer):
    """textcnn实现
    """

    def __init__(self, num_filters, emb_dim, kenral_size, dropout, class_num, **kwargs):
        self.emb_dim = emb_dim
        self.kenral_size = kenral_size
        self.dropout = dropout
        self.class_num = class_num
        self.num_filters = num_filters

        super(TextCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TextCNN, self).build(input_shape)
        self.conv2ds = [
            tf.keras.layers.Conv2D(self.num_filters, (n, self.emb_dim), padding="valid",
                                   activation=tf.keras.activations.relu)
            for n in self.kenral_size]
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        self.fc1 = tf.keras.layers.Dense(self.class_num)

    def call(self, inputs):
        x = inputs
        x = tf.expand_dims(x, axis=-1)
        x = [tf.nn.relu(conv(x)) for conv in self.conv2ds]

        maxpool_0 = tf.keras.layers.MaxPool2D(pool_size=(maxlen - 3 + 1, 1), strides=(1, 1), padding='valid')(x[0])
        maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(maxlen - 4 + 1, 1), strides=(1, 1), padding='valid')(x[1])
        maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(maxlen - 5 + 1, 1), strides=(1, 1), padding='valid')(x[2])

        x = tf.concat([maxpool_0, maxpool_1, maxpool_2], axis=1)
        x = tf.reshape(x, [-1, 3 * self.num_filters])
        x = self.dropout(x)
        print(x)

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
output = TextCNN(10, 128, [3, 4, 5], 0.8, 2)(embeddings)
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
