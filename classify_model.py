#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os

import datasets
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.utils.versions import require_version

from common.data_utils import convert_into_dataset_instance
from common.config import MrcConfig, RelConfig, DocSplitConfig

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`

class ClassifyModel(object):
    def __init__(self, config):
        self.args = config

    def load_model(self):
	    # Make one log on every process with the configuration for debugging.
	    logging.basicConfig(
	        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	        datefmt="%m/%d/%Y %H:%M:%S",
	        level=logging.INFO,
	    )
	
	    # If passed along, set the training seed now.
	    self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
	    self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)
	    self.model = AutoModelForSequenceClassification.from_pretrained(
	            self.args.model_name_or_path,
	            from_tf=bool(".ckpt" in self.args.model_name_or_path),
	            config=self.config,
	        )
	
	    # Preprocessing the datasets.
	    # Preprocessing is slighlty different for training and evaluation.
	
	    # Padding side determines if we do (question|context) or (context|question).
	    self.pad_on_right = self.tokenizer.padding_side == "right"
	
	    if self.args.max_seq_length > self.tokenizer.model_max_length:
	        logger.warning(
	            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
	            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
	        )
	
	    self.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)
	
	    # Preprocessing the datasets
	    self.sentence1_key, self.sentence2_key = "question", "context"
	
	    self.padding = "max_length" if self.args.pad_to_max_length else False

    def preprocess_function(self, examples):
        # Tokenize the texts
        texts = (
            (examples[self.args.question_column_name],) if self.args.context_column_name is None else (examples[self.args.question_column_name], examples[self.args.context_column_name])
        )
        result = self.tokenizer(*texts, padding=self.padding, max_length=self.args.max_seq_length, truncation=True)

        return result

    def predict(self, question, context):
	    # convert predict raw data into examples
	    # in stage of inference , we input one (question, context) pair into model each time
	    input_examples = convert_into_dataset_instance(question, context, self.args.question_column_name, self.args.context_column_name, self.args.id_name)
	    column_names = input_examples.column_names
	
	    # Predict Feature Creation
	    input_features = input_examples.map(
	        self.preprocess_function,
	        batched=True,
	        remove_columns=column_names,
	        desc="Running tokenizer on prediction dataset",
	    )
	    logger.info("input features : ")
	    logger.info(input_features)
	    # we dont need do anything on `input_features`
	    input_features_for_model = input_features
	    # DataLoaders creation:
	    if self.args.pad_to_max_length:
	        data_collator = default_data_collator
	    else:
	        data_collator = DataCollatorWithPadding(tokenizer)
	    predict_dataloader = DataLoader(
	        input_features_for_model, collate_fn=data_collator, batch_size=self.args.per_device_eval_batch_size
	        )
	    logger.info("input_features_for_model : ")
	    logger.info(input_features_for_model)
	
	    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
	    # shorter in multiprocess)
	
	    # Prediction
	    scores = []
	    for step, batch in enumerate(predict_dataloader):
	        outputs = self.model(**batch)
	        #predictions = outputs.logits.argmax(dim=-1)
	        logits = outputs.logits.detach().numpy()
	        for logit in logits:
	            exp_logit = np.exp(logit - np.max(logit))
	            probs = exp_logit / exp_logit.sum()
	            scores.append(probs[1])
	    return scores            

if __name__ == "__main__":
    rel_config_fpath = "/raid/loulianzhang/model/cross-lingual-MRC/rel_config.json"
    rel_config = RelConfig(rel_config_fpath)
    rel_model = ClassifyModel(rel_config)
    rel_model.load_model()
    question = ["Panthers đã mất bao nhiêu điểm trong phòng thủ?"]
    context = ["黑豹队的防守只丢了 308分，在联赛中排名第六，同时也以 24 次拦截领先国家橄榄球联盟 (NFL)，并且四次入选职业碗。职业碗防守截锋卡万·肖特以 11 分领先于全队，同时还有三次迫使掉球和两次重新接球。他的队友马里奥·爱迪生贡献了 6½ 次擒杀。黑豹队的防线上有经验丰富的防守端锋贾里德·艾伦，他是五次职业碗选手，曾以 136 次擒杀成为 NFL 职业生涯中的活跃领袖。另外还有在 9 场首发中就拿下 5 次擒杀的防守端锋科尼·伊利。在他们身后，黑豹队的三名首发线卫中有两人入选了职业碗：托马斯·戴维斯和卢克·坎克利。戴维斯完成了 5½ 次擒杀、四次迫使掉球和四次拦截，而坎克利带领球队在擒抱 (118) 中迫使两次掉球并拦截了他自己的四次传球。卡罗莱纳的第二防线有职业碗安全卫科特·科尔曼和职业碗角卫约什·诺曼，科尔曼带领球队完成了职业生涯中高达七次拦截并同时贡献了 88 次擒抱，而诺曼在本赛季成长为一名封锁角卫并完成了四次拦截，其中两次被判触地得分。"]
    rel_pred = rel_model.predict(question , context)
    print(rel_pred)
