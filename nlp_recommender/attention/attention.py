# -*- coding:utf-8 -*-
# @Time : 2021/9/17 10:42 下午
# @Author : huichuan LI
# @File : attention.py
# @Software: PyCharm

from tensorflow.keras.layers import *

'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''


def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12


def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = tf.shape(x)[-1]
    seq_len = tf.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = tf.keras.backend.temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = tf.concatenate(xs, 2)
    return tf.reshape(x, (-1, seq_len, kernel_size, seq_dim))


class Attention(Layer):
    """多头注意力机制
    """

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = tf.keras.layers.Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = tf.keras.layers.Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = tf.keras.layers.Dense(self.out_dim, use_bias=False)

    def call(self, inputs):
        q, k, v = inputs[: 3]

        V_len, Q_len = None, None
        # 这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
        if len(inputs) > 3:
            V_len = inputs[3]
            if len(inputs) > 4:
                Q_len = inputs[4]
        # 线性变换
        qw = self.q_dense(q)
        qw = tf.reshape(qw, (-1, tf.shape(qw)[1], self.heads, self.key_size))
        qw = tf.transpose(qw, [0, 2, 1, 3])

        kw = self.k_dense(k)
        kw = tf.reshape(kw, (-1, tf.shape(kw)[1], self.heads, self.key_size))
        kw = tf.transpose(kw, [0, 2, 1, 3])

        vw = self.v_dense(v)
        vw = tf.reshape(vw, (-1, tf.shape(vw)[1], self.heads, self.key_size))
        vw = tf.transpose(vw, [0, 2, 1, 3])
        # Attention

        A = tf.matmul(qw, kw, transpose_b=True) / tf.sqrt(float(self.key_size))
        A = tf.transpose(A, [0, 3, 2, 1])
        A = Mask(A, V_len, mode='add')
        A = tf.transpose(A, [0, 3, 2, 1])
        A = tf.nn.softmax(A)
        # 输出并mask
        O = tf.matmul(A, vw)
        O = tf.transpose(O, [0, 2, 1, 3])
        O = tf.reshape(O, (-1, tf.shape(O)[1], self.heads * self.key_size))
        O = Mask(O, Q_len, 'mul')
        return O

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class AtrousSelfAttention(Layer):
    """空洞多头自注意力机制
    说明：每个元素只跟相对距离为rate的倍数的元素有关联。
    """

    def __init__(self, heads, size_per_head, seq_len, seq_dim, rate=1,
                 key_size=None, mask_right=False, **kwargs):
        super(AtrousSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.rate = rate
        self.mask_right = mask_right
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            self.mask_right
        )

    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None

        pad_len = self.rate - self.seq_len % self.rate
        x = tf.keras.backend.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = tf.keras.backend.temporal_padding(x_mask, (0, pad_len))

        new_seq_len = tf.shape(x)[1]
        # 变换shape
        x = tf.reshape(x, (-1, new_seq_len // self.rate, self.rate, self.seq_dim))
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (-1, new_seq_len // self.rate, self.seq_dim))
        if x_mask is not None:
            x_mask = tf.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1))
            x_mask = tf.transpose(x_mask, (0, 2, 1, 3))
            x_mask = tf.reshape(x_mask, (-1, new_seq_len // self.rate, 1))
        # 做attention
        if x_mask is not None:
            x = self.attention([x, x, x, x_mask, x_mask])
        else:
            x = self.attention([x, x, x])
        # 恢复shape
        x = tf.reshape(x, (-1, self.rate, new_seq_len // self.rate, self.out_dim))
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (-1, new_seq_len, self.out_dim))
        x = x[:, : - pad_len]
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

S_inputs = tf.keras.Input(shape=(None,), dtype='int32')
embeddings = tf.keras.layers.Embedding(max_features, 128)(S_inputs)
# embeddings = SinCosPositionEmbedding(128)(embeddings) # 增加Position_Embedding能轻微提高准确率
O_seq = AtrousSelfAttention(8, 16, maxlen, 128)(embeddings)
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
