# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
@Author      :   luoyuankai 
@Time        :   2021/12/15 18:27:37
"""
import tensorflow as tf

from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:

    def __init__(self, hp):
        self.hp = hp
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)
        self.embedding_hour = tf.get_variable("embedding_hour", [168, self.hp.d_model // 4], dtype=tf.float32)
        self.embedding_minutes = tf.get_variable("embedding_minutes", [10000, self.hp.d_model // 4], dtype=tf.float32)
        self.embedding_lat = tf.get_variable("embedding_lat", [1700, self.hp.d_model // 4], dtype=tf.float32,
                                                trainable=True)
        self.embedding_lon = tf.get_variable("embedding_lon", [2100, self.hp.d_model // 4], dtype=tf.float32,
                                                trainable=True)
    def encode(self, xs, ys, mask, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens = ys
            input_hour, input_minutes, input_lat, input_lon, seqlens = xs

            # src_masks
            src_masks = tf.math.equal(input_lat, 0) # (N, T1)
            src_masks = tf.Print(src_masks,[src_masks,tf.shape(src_masks)],message= "mask:")
            inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, input_hour)
            inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, input_minutes)
            inputs_lon = tf.nn.embedding_lookup(self.embedding_lon, input_lon)
            inputs_lat = tf.nn.embedding_lookup(self.embedding_lat, input_lat)
            # embedding
            enc = tf.concat([inputs_hour, inputs_minutes, inputs_lat, inputs_lon], 2)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
            # Final linear projection (embedding weights are shared)
            outputs = tf.reshape(enc, [-1, self.hp.d_model])
            softmax_w = tf.get_variable("softmax_w", [self.hp.d_model, self.hp.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [self.hp.vocab_size], dtype=tf.float32)
            output = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
            y_hat = tf.to_int32(tf.argmax(output, axis=-1))
            return output, y_hat, y
            # return memory, src_masks

    # def decode(self, ys, xs, mask, memory, src_masks, training=True):
    #     '''
    #     memory: encoder outputs. (N, T1, d_model)
    #     src_masks: (N, T1)

    #     Returns
    #     logits: (N, T2, V). float32.
    #     y_hat: (N, T2). int32
    #     y: (N, T2). int32
    #     sents2: (N,). string.
    #     '''
    #     with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    #         decoder_inputs, y, seqlens = ys
    #         input_hour, input_minutes, input_lat, input_lon, seqlens = xs
    #         inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, input_hour)
    #         inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, input_minutes)
    #         inputs_lon = tf.nn.embedding_lookup(self.embedding_lon, input_lon)
    #         inputs_lat = tf.nn.embedding_lookup(self.embedding_lat, input_lat)
    #         # embedding
    #         dec = tf.concat([inputs_hour, inputs_minutes, inputs_lat, inputs_lon], 2)
    #         # tgt_masks
    #         dec *= self.hp.d_model ** 0.5  # scale

    #         tgt_masks = tf.math.equal(mask, 0)  # (N, T2)

    #         tgt_masks = tf.Print(tgt_masks,[tgt_masks,tf.shape(tgt_masks)],message= "mask:")
        
    #         dec += positional_encoding(dec, self.hp.maxlen2)
    #         dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

    #         # Blocks
    #         for i in range(self.hp.num_blocks):
    #             with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
    #                 # Masked self-attention (Note that causality is True at this time)
    #                 dec = multihead_attention(queries=dec,
    #                                           keys=dec,
    #                                           values=dec,
    #                                           key_masks=tgt_masks,
    #                                           num_heads=self.hp.num_heads,
    #                                           dropout_rate=self.hp.dropout_rate,
    #                                           training=training,
    #                                           causality=True,
    #                                           scope="self_attention")

    #                 # Vanilla attention
    #                 dec = multihead_attention(queries=dec,
    #                                           keys=memory,
    #                                           values=memory,
    #                                           key_masks=src_masks,
    #                                           num_heads=self.hp.num_heads,
    #                                           dropout_rate=self.hp.dropout_rate,
    #                                           training=training,
    #                                           causality=False,
    #                                           scope="vanilla_attention")
    #                 ### Feed Forward
    #                 dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

    #         # Final linear projection (embedding weights are shared)
    #         outputs = tf.reshape(dec, [-1, self.hp.d_model])
    #         softmax_w = tf.get_variable("softmax_w", [self.hp.d_model, self.hp.vocab_size], dtype=tf.float32)
    #         softmax_b = tf.get_variable("softmax_b", [self.hp.vocab_size], dtype=tf.float32)
    #         output = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
    #         y_hat = tf.to_int32(tf.argmax(output, axis=-1))

    #     return output, y_hat, y

    def train(self, xs, ys, mask):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        # memory, src_masks = self.encode(xs, mask)
        # logits, preds, y = self.decode(ys, xs, mask, memory, src_masks)
        logits, preds, y = self.encode(xs, ys, mask)
        input_hour_, input_minutes_, input_lat_, input_lon_, seqlens_ = xs

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        size = self.hp.vocab_size
        logits = tf.reshape(logits, [-1, size])
        logits = tf.Print(logits,[logits,tf.shape(logits)],message= "logits:")
        y_ = tf.reshape(y_, [-1, size])
        y_ = tf.Print(y_,[y_,tf.shape(y_)],message= "y_")
        mask = tf.reshape(mask, [-1])
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
#        nonpadding = tf.to_float(tf.not_equal(input_hour_, 0))
#        nonpadding = tf.reshape(nonpadding, [-1])
        nonpadding = tf.cast(mask, tf.float32)
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries, y, logits, mask

    def eval(self, xs, ys, mask):
        
        # memory, src_masks = self.encode(xs, mask, False)
        # logits, y_hat, y = self.decode(ys, xs, mask, memory, src_masks, False)
        logits, preds, y = self.encode(xs, ys, mask)
        summaries = tf.summary.merge_all()

        return logits, y, mask, summaries

