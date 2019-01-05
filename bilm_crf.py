"""BERT + bi-LSTM + CRF"""


__author__ = "Chenyang Yuan"


import tensorflow as tf
from tf_metrics import precision, recall, f1

import sys



class bilm_crf(object):
    """bilm+crf的实体词识别，在 https://github.com/guillaumegenthial/tf_ner 的基础上
    对接入BERT输出做了一些修改.
    
    Args:
        embeddings: embedding tensor from BERT
        labels: labels correspond to the input sequence with shape `[max_seq_length, batch_size]`
        mode: `tf.estimator.ModeKeys`
        seq_length_vec: vector contains the information of the input's length, e.g: `[l_1, l_2,...,l_n]`,n lessthan `max_seq_length`, l_i = `seq_length` """
    def __init__(self, embeddings, labels, number_labels, mode, seq_length_vec, lstm_size):
        self.embeddings = embeddings
        self.labels = labels
        self.number_labels = number_labels
        self.mode = mode
        self.seq_length_vec = seq_length_vec # a vector contains the information of the input feature`s length, ''
        self.lstm_size = lstm_size

    def graph_fn(self, reuse=None, getter=None):
        training = (self.mode == tf.estimator.ModeKeys.TRAIN)
        with tf.variable_scope('graph', reuse=None, custom_getter=getter):
            tf.summary.histogram("bert_output", self.embeddings)
            embeddings = tf.layers.dropout(self.embeddings, rate=0.5, training=False)
            # tf.Variable(embeddings, "bilm_embedding")
            # LSTM
            seq_length_vec = tf.transpose(self.seq_length_vec)
            hidden_size = embeddings.shape[-1].value
            t = tf.reshape(embeddings, [-1, 32, hidden_size])
            # t = tf.transpose(t, perm=[1, 0, 2])
            tf.logging.info("****seq_length-shape****")
            tf.logging.info(seq_length_vec.shape)
            tf.logging.info("****t-shape****")
            tf.logging.info(t.shape)
            tf.summary.histogram("lstm_input", t)
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.lstm_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.lstm_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=seq_length_vec)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=seq_length_vec)
            output = tf.concat([output_fw, output_bw], axis=-1)
            tf.summary.histogram("lstm_output", output)
            output = tf.transpose(output, perm=[1, 0, 2])
            output = tf.layers.dropout(output, rate=0.5, training=training)
            # tf.Variable(output, "bilm_output")
            # CRF
            tf.summary.histogram("lstm_output", output)
            logits = tf.layers.dense(output, self.number_labels)
            # tf.Variable(logits,"bilm_logits")
            crf_params = tf.get_variable("crf", [self.number_labels, self.number_labels], dtype=tf.float32) # 在计算图中创建变量

        return logits, crf_params

    def model_fn(self):
        logits, crf_params = self.graph_fn(self)
        tf.summary.histogram("logits", logits)
        seq_length_vec = tf.transpose(self.seq_length_vec)
        pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, seq_length_vec)
        variables = tf.get_collection("trainable_variables", "graph")
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            pass
        else:
            labels = self.labels
            logits = tf.reshape(logits, [32, 50, 5])
            labels = tf.reshape(labels, [32, 50])
            # seq_length_vec = tf.reshape(seq_length_vec, [32])
            crf_params = tf.reshape(crf_params, [5, 5])
            tf.logging.info("**********crf_log_likelihood input shape**********")
            tf.logging.info(logits.shape)
            tf.logging.info(labels.shape)
            tf.logging.info(seq_length_vec.shape)
            tf.logging.info(crf_params.shape)
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels, seq_length_vec, crf_params)
            # tf.summary.scalar("log_likelihood", log_likelihood)
            loss = tf.reduce_mean(-log_likelihood)
            indices = [i for i in range(self.number_labels)]
            weights = tf.sequence_mask(seq_length_vec, maxlen = 50)
            metrics = {
                'acc': tf.metrics.accuracy(labels, pred_ids, weights),
                'pr': tf.metrics.precision(labels, weights),
                'rc': tf.metrics.recall(labels, pred_ids, weights),
                # 'f1': tf.metrics.(labels, pred_ids, weights),
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])

            if self.mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    self.mode, loss=loss, eval_metric_ops=metrics)

            elif self.mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.train.AdamOptimizer().minimize(
                    loss, global_step=tf.train.get_or_create_global_step(),
                    var_list=variables)
                train_op = tf.group([train_op])
                return tf.estimator.EstimatorSpec(
                    self.mode, loss=loss, train_op=train_op)
