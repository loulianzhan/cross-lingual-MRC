#!/usr/bin/env python
# coding=utf-8

"""
句子切分，按整句（句号）切分
"""

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
        #sentence = re.findall(hanzi.sentence, document)
        document = re.sub('([。！？\?])([^”’])', r"\1\n\2", document)  # 单字符断句符
        document = re.sub('(\.{6})([^”’])', r"\1\n\2", document)  # 英文省略号
        document = re.sub('(\…{2})([^”’])', r"\1\n\2", document)  # 中文省略号
        document = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', document)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        document = document.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        sentence = document.split("\n")
        num_sent = len(sentence)
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

    document = open("/raid/loulianzhang/MRC/temp.txt","r").read()

    print(document)
    result = model.split(document)
    for res in result:
        print(res)
    import numpy as np
    print(np.mean([len(x) for x in result]))

if __name__=="__main__":
    main()
