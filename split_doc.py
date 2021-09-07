#!/usr/bin/env python
# coding=utf-8

import os
import sys

import re
from zhon import hanzi

class DocSplitModel(object):
    def __init__(self, config):
        self.args = config
        self.passage_window_length = self.args.passage_window_length
        self.sentence_stride = self.args.sentence_stride

    def split(self, document):
        sentence = re.findall(hanzi.sentence, document)
        num_sent = len(sentence)
        print(num_sent)
        # 滑窗切分成passage，以句子为单位步长
        if num_sent <= self.passage_window_length:
            return "".join(sentence)
        passage_list = []

        i = 0
        while i < num_sent - self.passage_window_length:
            curr_passage = sentence[i : i+self.passage_window_length]
            i += self.sentence_stride

            if curr_passage:
                passage_list.append("".join(curr_passage))

        return passage_list

class Config(object):
    def __init__(self):
        self.passage_window_length = 5
        self.sentence_stride = 2

def main():

    config = Config()
    model = DocSplitModel(config)

    document = open("temp.txt","r").read()

    result = model.split(document)
    for res in result:
        print(res)
    import numpy as np
    print(np.mean([len(x) for x in result]))

if __name__=="__main__":
    main()
