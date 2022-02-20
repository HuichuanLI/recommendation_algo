# -*- coding:utf-8 -*-
# @Time : 2021/9/21 5:28 下午
# @Author : huichuan LI
# @File : gpt.py
# @Software: PyCharm
# [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

import tensorflow as tf
from tensorflow import keras
import time
from transformer import Encoder
import pickle
import os
import utils


class GPT(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_len = max_len

        # I think task emb is not necessary for pretraining,
        # because the aim of all tasks is to train a universal sentence embedding
        # the body encoder is the same across all tasks,
        # and different output layer defines different task just like transfer learning.
        # finetuning replaces output layer and leaves the body encoder unchanged.

        # self.task_emb = keras.layers.Embedding(
        #     input_dim=n_task, output_dim=model_dim,  # [n_task, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )

        self.word_emb = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        # self.segment_emb = keras.layers.Embedding(
        #     input_dim=max_seg, output_dim=model_dim,  # [max_seg, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )
        self.position_emb = self.add_weight(
            name="pos", shape=[1, max_len, model_dim], dtype=tf.float32,  # [1, step, dim]
            initializer=keras.initializers.RandomNormal(0., 0.01))
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.task_mlm = keras.layers.Dense(n_vocab)
        self.task_nsp = keras.layers.Dense(2)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(lr)

    def call(self, seqs, training=False):
        embed = self.input_emb(seqs)  # [n, step, dim]
        z = self.encoder(embed, training=training, mask=self.mask(seqs))  # [n, step, dim]
        mlm_logits = self.task_mlm(z)  # [n, step, n_vocab]
        nsp_logits = self.task_nsp(tf.reshape(z, [z.shape[0], -1]))  # [n, n_cls]
        return mlm_logits, nsp_logits

    def step(self, seqs, seqs_, nsp_labels):
        with tf.GradientTape() as tape:
            mlm_logits, nsp_logits = self.call(seqs, training=True)
            pad_mask = tf.math.not_equal(seqs_, self.padding_idx)
            pred_loss = tf.reduce_mean(tf.boolean_mask(self.cross_entropy(seqs_, mlm_logits), pad_mask))
            nsp_loss = tf.reduce_mean(self.cross_entropy(nsp_labels, nsp_logits))
            loss = pred_loss + 0.2 * nsp_loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, mlm_logits

    def input_emb(self, seqs):
        return self.word_emb(seqs) + self.position_emb  # [n, step, dim]

    def mask(self, seqs):
        """
         abcd--
        a011111
        b001111
        c000111
        d000011
        -000011
        -000011
        force head not to see afterward. eg.
        a is a embedding for a---
        b is a embedding for ab--
        c is a embedding for abc-
        later, b embedding will + b another embedding from previous residual input to predict c
        """
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        pad = tf.math.equal(seqs, self.padding_idx)
        mask = tf.where(pad[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask  # (step, step)

    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.numpy() for l in self.encoder.ls],
        }
        return attentions


def train(model, data, step=10000, name="gpt"):
    t0 = time.time()
    for t in range(step):
        seqs, xlen, nsp_labels = data.sample(16)
        loss, pred = model.step(seqs[:, :-1], seqs[:, 1:], nsp_labels)
        if t % 100 == 0:
            pred = pred[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i] for i in seqs[0, 1:][:xlen[0].sum() + 1]]),
                "\n| prd: ", " ".join([data.i2v[i] for i in pred[:xlen[0].sum() + 1]]),
            )
            t0 = t1
    # os.makedirs("./visual/models/%s" % name, exist_ok=True)
    # model.save_weights("./visual/models/%s/model.ckpt" % name)


def export_attention(model, data, name="gpt"):
    model.load_weights("./visual/models/%s/model.ckpt" % name)

    # save attention matrix for visualization
    seqs, segs, xlen, nsp_labels = data.sample(32)
    model.call(seqs[:, :-1], segs[:, :-1], False)
    data = {"src": [[data.i2v[i] for i in seqs[j]] for j in range(len(seqs))], "attentions": model.attentions}
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4

    # d = utils.MRPCData("./MRPC", 2000)
    # print("num word: ", d.num_word)

    max_features = 20000
    maxlen = 80

    m = GPT(
        model_dim=MODEL_DIM, max_len=maxlen - 1, n_layer=N_LAYER, n_head=4, n_vocab=max_features,
        lr=LEARNING_RATE, drop_rate=0.2, padding_idx=0)
    # print(m.mask(d.sample(64)))
    # seqs, segs, xlen, nsp_labels = d.sample(32)
    # print(seqs)
    # print(segs)
    # print(xlen)
    # print(nsp_labels)

    from keras.preprocessing import sequence
    from keras.datasets import imdb
    import tensorflow as tf

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features, maxlen=maxlen
    )
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    t0 = time.time()
    step = 5000
    batch_size = 64
    for t in range(step):
        batches = len(x_train) // batch_size
        for b in range(batches):
            loss, pred = m.step(x_train[b * batch_size:b * batch_size + batch_size, :-1],
                                x_train[b * batch_size:b * batch_size + batch_size, 1:],
                                y_train[b * batch_size:b * batch_size + batch_size])
    if t % 100 == 0:
        pred = pred[0].numpy().argmax(axis=1)
        t1 = time.time()
        print(
            "\n\nstep: ", t,
            "| time: %.2f" % (t1 - t0),
            "| loss: %.3f" % loss.numpy()
        )
        t0 = t1
# train(m, d, step=5000, name="gpt")
# export_attention(m, d, name="gpt")
