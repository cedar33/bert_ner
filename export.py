"""Export model as a saved_model"""

__author__ = "Chenyang Yuan"

from pathlib import Path
import json

import tensorflow as tf

from bert_ner import model_fn_builder
import modeling

# DATADIR = '../../data/example'
# PARAMS = './results/params.json'
MODELDIR = '/home/ycy/workingdir/bert0.1/bert/output/result_dir'

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    # feature = InputFeatures(
    # input_ids=input_ids,
    # input_mask=input_mask,
    # segment_ids=segment_ids,
    # label_ids=label_id,
    # seq_length = seq_length,
    # is_real_example=True)

    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
    label_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='label_ids')
    seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_length')
    is_real_example = tf.placeholder(dtype=tf.string, shape=[None], name='is_real_example')
    receiver_tensors = {'input_ids': input_ids,
                            'input_mask': input_mask,
                            'segment_ids': segment_ids,
                            'label_ids':label_ids,
                            'seq_length':seq_length,
                            'is_real_example':is_real_example}
    features = {'input_ids': input_ids,
                            'input_mask': input_mask,
                            'segment_ids': segment_ids,
                            'label_ids':label_ids,
                            'seq_length':seq_length,
                            'is_real_example':is_real_example}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    bert_config = modeling.BertConfig.from_json_file("/home/ycy/workingdir/bert0.1/chinese_L-12_H-768_A-12/bert_config.json")
    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint="/home/ycy/workingdir/bert0.1/chinese_L-12_H-768_A-12/bert_model.ckpt",
      learning_rate=0.001,
      num_train_steps=100000,
      num_warmup_steps=10000,
      use_tpu=False,
      use_one_hot_embeddings=False)
    params={
    "num_labels": 15,
    }
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=None,
    model_dir=MODELDIR,
    save_checkpoints_steps=1000,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=1000,
        num_shards=8,
        per_host_input_for_training=is_per_host))

    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=32,
      eval_batch_size=32,
      predict_batch_size=32,
      params=params)
    estimator._export_to_tpu = False
    estimator.export_savedmodel('ner', serving_input_receiver_fn)
