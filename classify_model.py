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
from datasets import load_dataset, load_metric,Dataset
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from common.data_utils import convert_into_dataset_instance


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.10.0.dev0")

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
    pad_on_right = self.tokenizer.padding_side == "right"

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
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = tokenizer(*texts, padding=self.padding, max_length=self.args.max_seq_length, truncation=True)

        return result

    def predict(self, question, context):
    # convert predict raw data into examples
    # in stage of inference , we input one (question, context) pair into model each time
    input_examples = convert_into_dataset_instance(question, context)
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
        input_features_for_model, collate_fn=data_collator, batch_size=self.args.per_device_predict_batch_size
        )
    logger.info("input_features_for_model : ")
    logger.info(input_features_for_model)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Prediction
    for step, batch in enumerate(predict_dataloader):
        outputs = self.model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
    return predictions

if __name__ == "__main__":
    main()
