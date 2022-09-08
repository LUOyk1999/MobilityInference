# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs
import os
from hparams import Hparams
import math
import time
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

def get_performance(output, vocab_size, targets, mask):
	logits = np.reshape(output, [-1, vocab_size])
	labels = np.reshape(targets, [-1])
	mask_ = (np.reshape(mask, [-1])).astype(np.float32)
	# mask_ = np.cast(mask_, np.float32)
	# mask_ = mask_.astype(np.float32)

	predictions = (np.argmax(logits, 1)).astype(np.int32)
	# stayonly_labels = np.scalar_mul(2, labels)
	stayonly_labels = 2 * labels
	travel_tensor = np.ones_like(np.reshape(mask, [-1]))
	stay_tensor = np.zeros_like(np.reshape(mask, [-1]))

	travel_labels = np.equal(labels, travel_tensor)
	stay_labels = np.equal(labels, stay_tensor)
	correct_prediction = np.equal(predictions, labels)
	correct_negative_prediction = np.equal(predictions, stayonly_labels)

	positive = np.sum(travel_labels.astype(np.float32) * mask_)
	negative = np.sum(stay_labels.astype(np.float32) * mask_)
	accuracy = np.sum(correct_prediction.astype(np.float32) * mask_)
	negative_accuracy = np.sum(correct_negative_prediction.astype(np.float32) * mask_)

	TP = accuracy - negative_accuracy
	TN = negative_accuracy
	FN = positive - TP
	FP = negative - TN
	return TP, TN, FN, FP

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)
start_time = time.time()
logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train,hp.batch_size,shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval,hp.eval_batch_size,shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys, mask = iter.get_next()
train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries, y_, pred_, mask_ = m.train(xs, ys, mask)
y_hat, y_eval, mask_eval, eval_summaries = m.eval(xs, ys, mask)
saver = tf.train.Saver()
logging.info("# Session")
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)
    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    TPs = 0.0
    TNs = 0.0
    FNs = 0.0
    FPs = 0.0
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary, _y, _pred, _mask = sess.run([train_op, global_step, train_summaries, y_, pred_, mask_])
        epoch = math.ceil(_gs / num_train_batches)

        TP, TN, FN, FP = get_performance(_pred, 2, _y, _mask)
        TPs += TP
        TNs += TN
        FNs += FN
        FPs += FP
        VPs = TPs / (TPs + FPs)  # Travel Precision
        VRs = TPs / (TPs + FNs)  # Travel Recall
        SPs = TNs / (TNs + FNs)  # Stay Precision
        SRs = TNs / (TNs + FPs)  # Stay Recall
        TOL = TPs + TNs + FNs + FPs
        ACC = (TPs + TNs) / TOL
        F1_S = 2 * SPs * SRs / (SPs + SRs)
        F1_V = 2 * VPs * VRs / (VPs + VRs)
        F1 = 2 * F1_S * F1_V / (F1_V + F1_S)
        print("train : VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f" %
                    ( VPs, VRs, SPs, SRs, ACC, F1_S, F1_V, F1))
        if _gs and _gs % (10*num_train_batches) == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss
            TPs = 0.0
            TNs = 0.0
            FNs = 0.0
            FPs = 0.0
            logging.info("# test evaluation")
            _ = sess.run(eval_init_op)
            print(1)
            for j in range(num_eval_batches):
                y_hat_, y_eval_, mask_eval_ = sess.run([y_hat, y_eval, mask_eval])
                TP, TN, FN, FP = get_performance(y_hat_, 2, y_eval_, mask_eval_)
                TPs += TP
                TNs += TN
                FNs += FN
                FPs += FP
                VPs = TPs / (TPs + FPs)  # Travel Precision
                VRs = TPs / (TPs + FNs)  # Travel Recall
                SPs = TNs / (TNs + FNs)  # Stay Precision
                SRs = TNs / (TNs + FPs)  # Stay Recall
                F1_S = 2 * SPs * SRs / (SPs + SRs)
                F1_V = 2 * VPs * VRs / (VPs + VRs)
                F1 = 2 * F1_S * F1_V / (F1_V + F1_S)
                TOL = TPs + TNs + FNs + FPs
                ACC = (TPs + TNs) / TOL
                print("test : VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f Time: %.3f" %
                    ( VPs, VRs, SPs, SRs, ACC, F1_S, F1_V, F1, time.time() - start_time))    
            logging.info("# write results")
            print_to_file(hp.logdir+"/record.txt", hp.train+"Epoch: %d test: VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f Time: %.3f" % (
									  epoch, VPs, VRs, SPs, SRs, ACC, F1_S, F1_V, F1, time.time() - start_time))
            ckpt_name = os.path.join(hp.logdir, '10k')
            saver.save(sess, ckpt_name, global_step=_gs)
            TPs = 0.0
            TNs = 0.0
            FNs = 0.0
            FPs = 0.0
            logging.info("# fall back to train mode")
            sess.run(train_init_op)


logging.info("Done")
