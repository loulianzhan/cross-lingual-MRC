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

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.utils.versions import require_version

from common.utils_qa import postprocess_qa_predictions
from common.data_utils import convert_into_dataset_instance, create_and_fill_np_array
from common.config import MrcConfig, RelConfig, DocSplitConfig

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using device:', device)

class MrcModel(object):
    def __init__(self, args):
        self.args = args

    def load_model(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.args.model_name_or_path,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
            )
    
        # Padding side determines if we do (question|context) or (context|question).
        self.pad_on_right = self.tokenizer.padding_side == "right"
    
        if self.args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
    
        self.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.args.question_column_name] = [q.lstrip() for q in examples[self.args.question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.args.question_column_name],
            examples[self.args.context_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.args.pad_to_max_length else False
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def predict(self, question, context):
        # convert predict raw data into examples
        # in stage of inference , we input one (question, context) pair into model each time
        input_examples = convert_into_dataset_instance(question, context, self.args.question_column_name, self.args.context_column_name, self.args.id_name)
        column_names = input_examples.column_names
    
        # Predict Feature Creation
        input_features = input_examples.map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on prediction dataset")
        logger.info("input_features : ")
        logger.info(input_features)
        input_features_for_model = input_features.remove_columns(["example_id", "offset_mapping"])
        # DataLoaders creation:
        if self.args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorWithPadding(self.tokenizer)
        predict_dataloader = DataLoader(
            input_features_for_model, collate_fn=data_collator, batch_size=self.args.per_device_eval_batch_size
            )
        logger.info("input_features_for_model :")
        logger.info(input_features_for_model)
    
        # Prediction
        all_start_logits = []
        all_end_logits = []
        self.model = self.model.to(device)
        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                attention_mask, input_ids = batch["attention_mask"].to(device), batch["input_ids"].to(device)
                outputs = self.model(attention_mask = attention_mask, input_ids = input_ids)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
    
            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, input_features, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, input_features, max_len)

        # post-process predition result
        outputs_numpy = (start_logits_concat, end_logits_concat)
        predictions = postprocess_qa_predictions(
                examples=input_examples,
                features=input_features,
                predictions=outputs_numpy,
                version_2_with_negative=self.args.version_2_with_negative,
                n_best_size=self.args.n_best_size,
                max_answer_length=self.args.max_answer_length,
                null_score_diff_threshold=self.args.null_score_diff_threshold
            )
        return predictions

if __name__ == "__main__":
    mrc_config_fpath = "/raid/loulianzhang/model/cross-lingual-MRC/mrc_config.json"
    mrc_config = MrcConfig(mrc_config_fpath)
    mrc_model = MrcModel(mrc_config)
    mrc_model.load_model()
    question = "Panthers đã mất bao nhiêu điểm trong phòng thủ?"
    context = ["黑豹队的防守只丢了 308分，在联赛中排名第六，同时也以 24 次拦截领先国家橄榄球联盟 (NFL)，并且四次入选职业碗。职业碗防守截锋卡万·肖特以 11 分领先于全队，同时还有三次迫使掉球和两次重新接球。他的队友马里奥·爱迪生贡献了 6½ 次擒杀。黑豹队的防线上有经验丰富的防守端锋贾里德·艾伦，他是五次职业碗选手，曾以 136 次擒杀成为 NFL 职业生涯中的活跃领袖。另外还有在 9 场首发中就拿下 5 次擒杀的防守端锋科尼·伊利。在他们身后，黑豹队的三名首发线卫中有两人入选了职业碗：托马斯·戴维斯和卢克·坎克利。戴维斯完成了 5½ 次擒杀、四次迫使掉球和四次拦截，而坎克利带领球队在擒抱 (118) 中迫使两次掉球并拦截了他自己的四次传球。卡罗莱纳的第二防线有职业碗安全卫科特·科尔曼和职业碗角卫约什·诺曼，科尔曼带领球队完成了职业生涯中高达七次拦截并同时贡献了 88 次擒抱，而诺曼在本赛季成长为一名封锁角卫并完成了四次拦截，其中两次被判触地得分。"]*2
    print(len(context[0]))
    mrc_pred = mrc_model.predict(question , context)
    print(mrc_pred)
