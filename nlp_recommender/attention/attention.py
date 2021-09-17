# -*- coding:utf-8 -*-
# @Time : 2021/9/17 10:42 下午
# @Author : huichuan LI
# @File : attention.py
# @Software: PyCharm

from tensorflow.keras.layers import *


def to_mask(x, mask, mode='mul'):
    """通用mask函数
    这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [tf.shape(x) for x in inputs]
            else:
                input_shape = tf.shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs


def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = K.int_shape(x)[-1]
    seq_len = K.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = K.temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = K.concatenate(xs, 2)
    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))


class Attention(OurLayer):
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
        v_mask, q_mask = None, None
        # 这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = tf.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = tf.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = tf.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = tf.transpose(qw, (0, 2, 1, 3))
        kw = tf.transpose(kw, (0, 2, 1, 3))
        vw = tf.transpose(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = to_mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right is True:
                ones = K.ones_like(a[: 1, : 1])
                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
                a = a - mask
            else:
                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]
                mask = (1 - K.constant(self.mask_right)) * 1e10
                mask = K.expand_dims(K.expand_dims(mask, 0), 0)
                self.mask = mask
                a = a - mask
        a = K.softmax(a)
        self.a = a
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = to_mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


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
O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
