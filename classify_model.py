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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using device:', device)

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
        self.model = self.model.to(device)
        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                attention_mask, input_ids, token_type_ids = batch["attention_mask"].to(device), batch["input_ids"].to(device), batch["token_type_ids"].to(device)
                outputs = self.model(attention_mask = attention_mask, input_ids = input_ids, token_type_ids=token_type_ids)
            #predictions = outputs.logits.argmax(dim=-1)
            logits = outputs.logits.cpu().numpy()
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
    question = "Tổng Bí thư Tập Cận Bình thị sát Làng Mao Đài Sơn vào tháng mấy?"
    context = ['眼下，广西桂林市全州县才湾镇毛竹山村，一串串成熟的葡萄挂满枝梢，到处弥漫着丰收的喜悦。“今年的葡萄还未上市就已经被各地顾客订购一空。”该村葡萄协会会长王海荣兴奋地说。从民谣里“有女不嫁毛竹山”的穷乡僻壤，到现在的“家家户户住楼房，共同富裕奔小康”，毛竹山村由“一串葡萄”引发的幸福蝶变已成为佳话。', '该村葡萄协会会长王海荣兴奋地说。从民谣里“有女不嫁毛竹山”的穷乡僻壤，到现在的“家家户户住楼房，共同富裕奔小康”，毛竹山村由“一串葡萄”引发的幸福蝶变已成为佳话。脱贫摘帽不是终点，而是新生活、新奋斗的起点。', '从民谣里“有女不嫁毛竹山”的穷乡僻壤，到现在的“家家户户住楼房，共同富裕奔小康”，毛竹山村由“一串葡萄”引发的幸福蝶变已成为佳话。脱贫摘帽不是终点，而是新生活、新奋斗的起点。今年4月，习近平总书记在毛竹山村考察时深情地说，让人民生活幸福是“国之大者”。', '脱贫摘帽不是终点，而是新生活、新奋斗的起点。今年4月，习近平总书记在毛竹山村考察时深情地说，让人民生活幸福是“国之大者”。全面推进乡村振兴的深度、广度、难度都不亚于脱贫攻坚，决不能有任何喘口气、歇歇脚的想法，要在新起点上接续奋斗，推动全体人民共同富裕取得更为明显的实质性进展。', '今年4月，习近平总书记在毛竹山村考察时深情地说，让人民生活幸福是“国之大者”。全面推进乡村振兴的深度、广度、难度都不亚于脱贫攻坚，决不能有任何喘口气、歇歇脚的想法，要在新起点上接续奋斗，推动全体人民共同富裕取得更为明显的实质性进展。牢记总书记的殷殷嘱托，广西笃定前行、乘势而上，全面推进巩固拓展脱贫攻坚成果同乡村振兴有效衔接，在新的起点上谱写乡村振兴新篇章。', '牢记总书记的殷殷嘱托，广西笃定前行、乘势而上，全面推进巩固拓展脱贫攻坚成果同乡村振兴有效衔接，在新的起点上谱写乡村振兴新篇章。摘帽不摘帮扶 乡村风貌换新颜“幸亏有政府19400元的临时救助，不然我家又要返贫了。”', '摘帽不摘帮扶 乡村风貌换新颜“幸亏有政府19400元的临时救助，不然我家又要返贫了。”家住贺州市平桂区鹅塘镇明梅村的李有才说。脱贫后，眼看日子慢慢好起来，李有才夫妻俩却先后生病入院，巨大的开销让他犯了愁。', '“幸亏有政府19400元的临时救助，不然我家又要返贫了。”家住贺州市平桂区鹅塘镇明梅村的李有才说。脱贫后，眼看日子慢慢好起来，李有才夫妻俩却先后生病入院，巨大的开销让他犯了愁。为了帮李有才一家渡过难关，帮扶干部及时为他们申请了临时救助。', '脱贫后，眼看日子慢慢好起来，李有才夫妻俩却先后生病入院，巨大的开销让他犯了愁。为了帮李有才一家渡过难关，帮扶干部及时为他们申请了临时救助。“我们常态化关注脱贫不稳定户，及时进行跟踪监测，一旦发现返贫风险，按照‘缺什么补什么’的原则落实帮扶政策。”平桂区负责开展防止返贫动态监测工作的干部杨昌作说。', '“我们常态化关注脱贫不稳定户，及时进行跟踪监测，一旦发现返贫风险，按照‘缺什么补什么’的原则落实帮扶政策。”平桂区负责开展防止返贫动态监测工作的干部杨昌作说。目前，平桂区累计识别监测对象1864户7722人，已解除风险累计721户3269人。', '平桂区负责开展防止返贫动态监测工作的干部杨昌作说。目前，平桂区累计识别监测对象1864户7722人，已解除风险累计721户3269人。除了动态监测，脱贫户该享受的政策一点也没少。', '除了动态监测，脱贫户该享受的政策一点也没少。在梧州岑溪市南渡镇西竹村，脱贫户唐尚宏今年养牛获得3000元的产业奖补，两个孩子持续享受教育保障优惠政策。', '在梧州岑溪市南渡镇西竹村，脱贫户唐尚宏今年养牛获得3000元的产业奖补，两个孩子持续享受教育保障优惠政策。“今年养牛进账2万多元，这日子越过越有奔头。”唐尚宏信心满满。', '在梧州岑溪市南渡镇西竹村，脱贫户唐尚宏今年养牛获得3000元的产业奖补，两个孩子持续享受教育保障优惠政策。“今年养牛进账2万多元，这日子越过越有奔头。”唐尚宏信心满满。按照中央要求，脱贫攻坚目标任务完成后，将设立5年过渡期。', '唐尚宏信心满满。按照中央要求，脱贫攻坚目标任务完成后，将设立5年过渡期。“过渡期内保持主要帮扶政策总体稳定，从集中资源支持脱贫攻坚转向巩固拓展脱贫攻坚成果和全面推进乡村振兴。”广西壮族自治区乡村振兴局党组副书记杨宏博介绍，广西严格落实“四个不摘”要求，并对现有帮扶政策进行分类优化调整，确保政策不断档。']
    rel_pred = rel_model.predict(question , context)
    print(rel_pred)
