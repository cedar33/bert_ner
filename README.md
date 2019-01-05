# bert_ner
### 1.理论概述
基于google提出的bert模型和tensorflow实现，bilm+crf部分参考了[Guillaume Genthial的代码](https://github.com/guillaumegenthial/tf_ner)，详细理论讲解可见 [右左瓜子的知乎专栏](https://zhuanlan.zhihu.com/p/52248160)
### 2.Requires
* google的bert实现[google research bert](https://github.com/google-research/bert)
* google提供的汉语预训练数据 [chinese-pretrain](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
### 3.使用方法
* 1.下载以上require中的代码和文件，将`bilm_crf.py`和`bert_bilm_crf.py`放到和google的`bert`中和`modeling.py`平级的文件夹下
* 2.运行脚本
```shell
python .\bert_bilm_crf.py --task_name=ner --do_train=true --do_eval=false --do_predict=false --data_dir=path\to\yourdata \
--vocab_file=path\to\chinese_L-12_H-768_A-12\vocab.txt --bert_config_file=path\to\chinese_L-12_H-768_A-12\bert_config.json \
--init_checkpoint=path\to\chinese_L-12_H-768_A-12\bert_model.ckpt --max_seq_length=50 --train_batch_size=32 \
--learning_rate=5e-5 --num_train_epochs=2.0 --output_dir=/tmp/ner_output/
```
### 4.关于训练数据
在给出的example.tsv中有两行示例数据，把格式整理成类似的即可
***
### 1.theory detail
this is a solution to NER task base on `BERT` and `bilm+crf`, the `BERT` model comes from [google's github](https://github.com/google-research/bert), the bilm+crf part inspired from [Guillaume Genthial's code](https://github.com/guillaumegenthial/tf_ner), visit [this page](https://zhuanlan.zhihu.com/p/52248160) for more details
### 2.Requires
* google's `BERT` model [google research bert](https://github.com/google-research/bert)
* a Chinese pre-trained model from google[chinese-pretrain](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
### 3.how to use
* 1.download codes and files mentioned above, put the `bilm_crf.py` and `bert_bilm_crf.py` into the google's `BERT` directory, the to python file is at the same level with `modeling.py`
* 2.run script
```shell
python .\bert_bilm_crf.py --task_name=ner --do_train=true --do_eval=false --do_predict=false --data_dir=path\to\yourdata \
--vocab_file=path\to\chinese_L-12_H-768_A-12\vocab.txt --bert_config_file=path\to\chinese_L-12_H-768_A-12\bert_config.json \
--init_checkpoint=path\to\chinese_L-12_H-768_A-12\bert_model.ckpt --max_seq_length=50 --train_batch_size=32 \
--learning_rate=5e-5 --num_train_epochs=2.0 --output_dir=/tmp/ner_output/
```
### 4.about trainning data
there is an example data in `example.tsv` to show the formate, you are surpoed to transform your data into this formate, or you can modify the `input_fn` in `bert_bilm_crf.py`
