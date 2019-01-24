import requests
import json

from bert_ner import convert_examples_to_features
from bert_ner import InputExample
from bert_ner import convert_single_example

import tokenization

NERLABEL = ["contradiction", "BPER", "BLOC", "BORG", "BTIME", "IPER", "ILOC", "IORG", "ITIME", "EPER", "ELOC", "EORG", "ETIME", "O", "OTHER"]


def nameEntity_recognition(line):
    vocab_file = "/home/ycy/workingdir/bert0.1/chinese_L-12_H-768_A-12/vocab.txt"
    tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=False)
    line = line.replace(" ","")
    print(line)
    example = InputExample(0, line, [])
    feature = convert_single_example(0, example, NERLABEL, 128, tokenizer)
    data = dict()
    # instance=dict()
    # words = line
    # # print(words)
    # instance["words"] = words
    # instance["nwords"] = len(words)
    data["instances"]=[feature2json(feature)]
    # print(data)
    # print(json.dumps(data).encode("utf-8"))
    headers = {"content-type": "application/json"}
    print(feature2json(feature))
    json_response = requests.post('http://172.16.20.228:8501/v1/models/ner:predict', data=json.dumps(data).encode("utf-8"), headers=headers)
    pred = json.loads(json_response.text)["predictions"][0]
    for w,p in zip(list(line), pred):
        print(w, NERLABEL[p])
    # tempwords = ""
    # for w,t in zip(words, predictions):
    #     if t == "O":
    #         resultlist.append(w)
    #     elif t=="B" or t == "I":
    #         tempwords += w
    #     elif t == "E":
    #         tempwords += w
    #         resultlist.append(tempwords)
    #         tempwords = ""
    # # print(" ".join(resultlist))
    # # print(resultlist)
    # instance["words"] = resultlist
    # instance["nwords"] = len(resultlist)
    # data["instances"]=[instance]
    # json_response = requests.post('http://172.16.20.228:9001/v1/models/ner:predict', data=json.dumps(data).encode("utf-8"), headers=headers)
    # # print(json.loads(json_response.text))
    # predictions = json.loads(json_response.text)['predictions'][0]["tags_ema"]
    # resultdict = {"PER":[],"LOC":[],"ORG":[],"TIME":[],"OTHER":[]}
    # # PER_list = list()
    # # LOC_list = list()
    # # TIME_list = list()
    # # ORG_list = list()
    # # OTHER_list = list()
    # tmpwords = ""
    # tmptag = ""
    # for w, t in zip(resultlist, predictions):
    #     if t[0] == "O":
    #         if len(tmpwords) != 0:
    #             resultdict[tmptag[1:]].append(tmpwords)
    #             tmpwords = ""
    #     elif t[0] == "B":
    #         tmpwords += w
    #     elif t[0] == "I":
    #         tmpwords += w
    #     elif t[0] == "E":
    #         tmpwords += w
    #         resultdict[t[1:]].append(tmpwords)
    #         tmpwords = ""
    #     tmptag = t
    # return resultdict
    return 0

def feature2json(obj):
    return {
            'input_ids': obj.input_ids,
            'input_mask': obj.input_mask,
            'segment_ids': obj.segment_ids,
            'label_ids':obj.label_ids,
            'seq_length':obj.seq_length,
            'is_real_example': "true"
            }

if __name__ == '__main__':
    line = "华为孟晚舟被逮捕"
    entity = nameEntity_recognition(line)
    print(entity)
