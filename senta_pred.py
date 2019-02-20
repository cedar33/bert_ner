"""Export model as a saved_model"""

__author__ = "Chenyang Yuan"

from pathlib import Path
import json
import re
import time

import tensorflow as tf
import functools
import tokenization
import numpy as np

from bert_senta import model_fn_builder
import modeling
import zhconv
from pymongo import MongoClient
import pymysql

# DATADIR = '../../data/example'
# PARAMS = './results/params.json'
MODELDIR = '/home/ycy/workingdir/bert0.1/bert/output/senta_dir'
vocab_file = "/home/ycy/workingdir/textdata/senta/chinese_L-12_H-768_A-12/vocab.txt"
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=False)
SENTALABEL = ["contradiction", "negative", "positive"]

port = 0
client = MongoClient('localhost', port)
db = client['twitter']
twitter_collection = db["tweets"]

def twitter_gen():
    # 查询数据库或者读取文件把文本转化为feature然后yeild出来
     yield feature

def twitter_id_gen(): # for update method, to tell method which twitter should be updated
    for twitter in   twitter_collection.find({"label":{"$exists":False}},{"tweet_text":1, "_id":1}, no_cursor_timeout=True):
        id = twitter["_id"]
        line = twitter["tweet_text"]
        yield id, line

def getkeyword():
#    通过关键词进一步细分，这一步通过查库或者读取文件返回关键词列表
    return keyword_list

def get_neg_type(keyword_list, line):
    # print(keyword_list)
    for keyword in keyword_list:
        if line.count(keyword[1]) > 0:
            return keyword[0]
    return "NEG"

def senta_update(id, label, line, keyword_list):
    '''Args:
        id: id in mongo,
        label: predict output,
        line: line after cleanning
        keyword_list: keyword_list in mysql'''
    # 更新文本情感分析结果
    return 0

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, labels=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                seq_length,
                is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example
        self.seq_length = seq_length


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0:(max_seq_length)]
    tokens = []
    segment_ids = []
    # tokens.append("[CLS]")
    # segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    # tokens.append("[SEP]")
    # segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    seq_length = len(input_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_id = [0, 0, 0]
    label_id[label_map[example.labels]] = 1

    feature = {
        "input_ids":input_ids,
        "input_mask":input_mask,
        "segment_ids":segment_ids,
        "label_ids": label_id,
        "seq_length" : seq_length,
        "is_real_example":1}
    return feature

def pred_infn(params=None):
    types = {"input_ids":tf.int32, "input_mask":tf.int32, "segment_ids":tf.int32, "label_ids":tf.int32, "seq_length":tf.int32, "is_real_example":tf.int32}
    shapes = {"input_ids":(128,), "input_mask":(128,), "segment_ids":(128,), "label_ids":(3), "seq_length":(), "is_real_example":()}
    dataset = tf.data.Dataset.from_generator(twitter_gen, output_types=types, output_shapes=shapes)
    dataset = (dataset.batch(1))
    return dataset


if __name__ == '__main__':
    bert_config = modeling.BertConfig.from_json_file("/home/ycy/workingdir/textdata/senta/chinese_L-12_H-768_A-12/bert_config.json")
    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint="/home/ycy/workingdir/textdata/senta/chinese_L-12_H-768_A-12/bert_model.ckpt",
      learning_rate=0.001,
      num_train_steps=100000,
      num_warmup_steps=10000,
      use_tpu=False,
      use_one_hot_embeddings=False)
    params={
    "num_labels": 3,
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
    pred = estimator.predict(input_fn=pred_infn)
    starttime = time.time()
    keyword_list = getkeyword()
    twitter_gen_2 = twitter_id_gen()
    counter = 0  
    for p in pred:
        # print(twitter_gen_2.__next__()[0], twitter_gen_2.__next__()[1])
        twitter_info = twitter_gen_2.__next__()
        line = twitter_info[1]
        line = zhconv.convert(re.sub(r'[^(\u4e00-\u9fa5|,|.|、|，|。|！|？|?|!)]', "", line, count=0, flags=0), "zh-cn")
        id = twitter_info[0]
        senta_update(id, p, line, keyword_list)
        counter += 1
        if counter % 10000 == 0:
            print(time.time()-starttime)
