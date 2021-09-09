#!/usr/bin/env python
# coding=utf-8
"""
基于文档提问的机器阅读理解模型的接口
"""

import json
import os

# 根据实际GPU情况选取
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from common.config import MrcConfig, RelConfig, DocSplitConfig
from mrc_model import MrcModel
from classify_model import ClassifyModel 
from split_doc import DocSplitModel
import numpy as np

class MrcBasedDocModel(object):
    # 初始化参数
    def __init__(self, model_dir):
        mrc_config_fpath = os.path.join(model_dir, "mrc_config.json")
        rel_config_fpath = os.path.join(model_dir, "rel_config.json")
        doc_split_config_fpath = os.path.join(model_dir, "doc_split_config.json")

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
        scores = self.classify_model.predict(question, passage)
        index_of_top_pred = np.argmax(scores)
        context = passage[index_of_top_pred]

        # MRC预测
        mrc_pred = self.mrc_model.predict(question, context)
        return mrc_pred

if __name__=="__main__":
   
    model_dir = "/raid/loulianzhang/model/cross-lingual-MRC/"
    qa_model = MrcBasedDocModel(model_dir)
    qa_model.load_model()
    # <限定passage模式>
    question = "Panthers đã mất bao nhiêu điểm trong phòng thủ?"
    context = "黑豹队的防守只丢了 308分，在联赛中排名第六，同时也以 24 次拦截领先国家橄榄球联盟 (NFL)，并且四次入选职业碗。职业碗防守截锋卡万·肖特以 11 分领先于全队，同时还有三次迫使掉球和两次重新接球。他的队友马里奥·爱迪生贡献了 6½ 次擒杀。黑豹队的防线上有经验丰富的防守端锋贾里德·艾伦，他是五次职业碗选手，曾以 136 次擒杀成为 NFL 职业生涯中的活跃领袖。另外还有在 9 场首发中就拿下 5 次擒杀的防守端锋科尼·伊利。在他们身后，黑豹队的三名首发线卫中有两人入选了职业碗：托马斯·戴维斯和卢克·坎克利。戴维斯完成了 5½ 次擒杀、四次迫使掉球和四次拦截，而坎克利带领球队在擒抱 (118) 中迫使两次掉球并拦截了他自己的四次传球。卡罗莱纳的第二防线有职业碗安全卫科特·科尔曼和职业碗角卫约什·诺曼，科尔曼带领球队完成了职业生涯中高达七次拦截并同时贡献了 88 次擒抱，而诺曼在本赛季成长为一名封锁角卫并完成了四次拦截，其中两次被判触地得分。"
    prediction = qa_model.passage_qa_predict(question , context)
    print(prediction)
    
    # <限定doc模式>
    prediction = qa_model.doc_qa_predict(question , context)
    print(prediction)
