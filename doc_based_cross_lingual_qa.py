#!/usr/bin/env python
# coding=utf-8
"""
基于文档提问的机器阅读理解模型的接口
"""

import json
import os
import sys
from common.config import MrcConfig, RelConfig, DocSplitConfig
from mrc_model import MrcModel
from classify_model import ClassifyModel 
from split_doc import DocSplitModel
import numpy as np

class MrcBasedDocModel(object):
    # 初始化参数
    def _init_(self, mrc_config_fpath, rel_config_fpath, doc_split_config_fpath):
        self.mrc_config = MrcConfig(mrc_config_fpath)
        self.rel_config = RelConfig(rel_config_fpath)
        self.doc_split_config = DocSplitConfig(doc_split_config_fpath)

    # 加载模型
    def load_model(self):
        # 加载MRC模型
        self.mrc_model = MrcModel(self.mrc_config)
        self.mrc_model.load_model()
        
        # 加载classifier模型
        self.classify_model = ClassifyModel(self.rel_config)
        self.classify_model.load_model()

        # 加载切分句子模型
        self.doc_split_model = DocSplitModel(self.doc_split_config)

    # <限定passage模式> 预测接口
    def passage_qa_predict(self, question , context):
        # MRC预测
        mrc_pred = self.mrc_model.predict(question , context)
        return mrc_pred

    # <限定doc模式> 预测接口
    def doc_qa_predict(self, question , document):
        # 切分长文档
        passage = self.doc_split_model.split(document)

        # 排序模型
        predictions = self.classify_model(question, passage)
        scores = [x[1] for x in predictions]
        index_of_top_pred = np.argmax(scores)
        context = passage[index_of_top_pred]

        # MRC预测
        mrc_pred = self.mrc_model.predict(question, context)
        return mrc_pred
