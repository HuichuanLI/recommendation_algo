# -*- coding:utf-8 -*-
# @Time : 2021/9/21 9:37 下午
# @Author : huichuan LI
# @File : bert.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import time
from gpt import GPT
import os
import pickle

from keras.preprocessing import sequence
from keras.datasets import imdb
import tensorflow as tf


class BERT(GPT):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__(model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg, drop_rate, padding_idx)
        # I think task emb is not necessary for pretraining,
        # because the aim of all tasks is to train a universal sentence embedding
        # the body encoder is the same across all tasks,
        # and different output layer defines different task just like transfer learning.
        # finetuning replaces output layer and leaves the body encoder unchanged.

        # self.task_emb = keras.layers.Embedding(
        #     input_dim=n_task, output_dim=model_dim,  # [n_task, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )

    def step(self, seqs, seqs_, loss_mask, nsp_labels):
        with tf.GradientTape() as tape:
            mlm_logits, nsp_logits = self.call(seqs, training=True)
            mlm_loss_batch = tf.boolean_mask(self.cross_entropy(seqs_, mlm_logits), loss_mask)
            mlm_loss = tf.reduce_mean(mlm_loss_batch)
            nsp_loss = tf.reduce_mean(self.cross_entropy(nsp_labels, nsp_logits))
            loss = mlm_loss + 0.2 * nsp_loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, mlm_logits

    def mask(self, seqs):
        mask = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # [n, 1, 1, step]


def _get_loss_mask(len_arange, seq, pad_id):
    rand_id = np.random.choice(len_arange, size=max(2, int(MASK_RATE * len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id, dtype=np.bool)
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id


def do_mask(seq, len_arange, pad_id, mask_id):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = mask_id
    return loss_mask


def do_replace(seq, len_arange, pad_id, word_ids):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = np.random.choice(word_ids, size=len(rand_id))
    return loss_mask


def do_nothing(seq, len_arange, pad_id):
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask


def random_mask_or_replace(seqs, xlen, nsp_labels, arange, pad_id, max_len, max_feature):
    seqs_ = seqs.copy()
    p = np.random.random()
    if p < 0.7:
        # mask
        loss_mask = np.concatenate(
            [do_mask(
                seqs[i, :],
                range(xlen[i]),
                pad_id, max_len + 1
            ) for i in range(len(seqs))], axis=0)
    elif p < 0.85:
        # do nothing
        loss_mask = np.concatenate(
            [do_nothing(
                seqs[i, :],
                range(xlen[i]),
                pad_id) for i in range(len(seqs))], axis=0)
    else:
        # replace
        loss_mask = np.concatenate(
            [do_replace(
                seqs[i, :],
                range(xlen[i]),
                pad_id,
                list(range(max_feature))) for i in range(len(seqs))], axis=0)
    return seqs, seqs_, loss_mask, xlen, nsp_labels


if __name__ == "__main__":
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4
    MASK_RATE = 0.15

    max_features = 20000
    maxlen = 80

    m = BERT(
        model_dim=MODEL_DIM, max_len=maxlen, n_layer=N_LAYER, n_head=4, n_vocab=max_features + 1,
        lr=LEARNING_RATE, drop_rate=0.2, padding_idx=0)

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features, maxlen=maxlen
    )
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Pad sequences (samples x time)')
    x_len_train = np.array([len(x) for x in x_train])
    x_len_test = np.array([len(x) for x in x_test])

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print(x_len_train)

    t0 = time.time()
    arange = np.arange(0, maxlen)
    step = 10000
    for t in range(step):
        rand_id = np.random.choice(range(len(x_train)), size=32, replace=False)

        seqs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(seqs=x_train[rand_id, :],
                                                                          xlen=x_len_train[rand_id],
                                                                          nsp_labels=y_train[rand_id],
                                                                          arange=arange,
                                                                          pad_id=0, max_len=maxlen,
                                                                          max_feature=max_features)
        loss, pred = m.step(seqs, seqs_, loss_mask, nsp_labels)
        if t % 100 == 0:
            pred = pred[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
            )
            t0 = t1
