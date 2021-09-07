"""
将包含所有预测模块相关参数的配置文件(*.json)
解析成Config类
"""

import os,sys
import json


def load_config(fpath):
    with open(fpath, "r") as fr:
        config = json.load(fr)
    return config

"""
整个模型的基本参数配置
"""
class BaseConfig(object):
    def __init__(self, config_fpath):
        config = load_config(config_fpath)
        self.relevance_topk = config["relevance_topk"]

"""
MRC模型推理接口相关的配置参数
"""
class MrcConfig(object):
    def __init__(self, config_fpath):
        config = load_config(config_fpath)
        self.question_column_name = config["question_column_name"]
        self.context_column_name = config["context_column_name"]
        self.id_name = config["id_name"]
        self.max_seq_length = config["max_seq_length"]
        self.pad_to_max_length = config["pad_to_max_length"]
        self.model_name_or_path = config["model_name_or_path"]
        self.per_device_eval_batch_size = config["per_device_eval_batch_size"]
        self.doc_stride = config["doc_stride"]
        self.n_best_size = config["n_best_size"]
        self.null_score_diff_threshold = config["null_score_diff_threshold"]
        self.version_2_with_negative = config["version_2_with_negative"]
        self.max_answer_length = config["max_answer_length"]

"""
relevance模型推理接口相关的配置参数
"""
class RelConfig(object):
    def __init__(self, config_fpath):
        config = load_config(config_fpath)
        self.question_column_name = config["question_column_name"]
        self.context_column_name = config["context_column_name"]
        self.id_name = config["id_name"]
        self.max_seq_length = config["max_seq_length"]
        self.pad_to_max_length = config["pad_to_max_length"]
        self.model_name_or_path = config["model_name_or_path"]
        self.per_device_eval_batch_size = config["per_device_eval_batch_size"]

"""
滑窗模型接口相关的配置参数
"""
class DocSplitConfig(object):
    def __init__(self, config_fpath):
        config = load_config(config_fpath)
        self.sentence_stride = config["sentence_stride"]
        self.passage_window_length = config["passage_window_length"]
