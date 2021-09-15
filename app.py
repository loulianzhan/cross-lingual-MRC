from sanic import Sanic
from sanic.response import json,text

# 请预先将模型放在同级目录下
model_dir = './model/'

import time
import logging
import os

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.utils.versions import require_version

from common.utils_qa import postprocess_qa_predictions
from common.data_utils import convert_into_dataset_instance, create_and_fill_np_array
from common.config import MrcConfig, RelConfig, DocSplitConfig
from split_doc import DocSplitModel

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:7"
logger.info('Using device:', device)

"""
定义MRC模型
"""
class MrcModel(object):
    def __init__(self, args):
        self.args = args

    def load_model(self):
        logger.info(f"-------model_dir : {self.args.model_name_or_path}")
    
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.args.model_name_or_path,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
            )
        self.model = self.model.to(device)
        logger.info(f"Filish loading MRC model , GPU device {device}")
    
        self.pad_on_right = self.tokenizer.padding_side == "right"
    
        if self.args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
    
        self.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(self, examples):
        examples[self.args.question_column_name] = [q.lstrip() for q in examples[self.args.question_column_name]]

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

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    def predict(self, question, context):
        start = time.time()
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
        end = time.time()
        return predictions

"""
定义排序模型
"""
class ClassifyModel(object):
    def __init__(self, config):
        self.args = config

    def load_model(self):
        # If passed along, set the training seed now.
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
            )
        self.model = self.model.to(device)
        logger.info(f"Filish loading rank model , GPU device {device}")
    
        self.pad_on_right = self.tokenizer.padding_side == "right"
    
        if self.args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
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
    
        # Prediction
        scores = []
        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                attention_mask, input_ids, token_type_ids = batch["attention_mask"].to(device), batch["input_ids"].to(device), batch["token_type_ids"].to(device)
                outputs = self.model(attention_mask = attention_mask, input_ids = input_ids, token_type_ids=token_type_ids)
            logits = outputs.logits.cpu().numpy()
            for logit in logits:
                exp_logit = np.exp(logit - np.max(logit))
                probs = exp_logit / exp_logit.sum()
                scores.append(probs[1])
        return scores            

mrc_config_fpath = os.path.join(model_dir, "mrc_config.json")
rel_config_fpath = os.path.join(model_dir, "rel_config.json")
doc_split_config_fpath = os.path.join(model_dir, "doc_split_config.json")

mrc_config = MrcConfig(mrc_config_fpath)
rel_config = RelConfig(rel_config_fpath)
doc_split_config = DocSplitConfig(doc_split_config_fpath)

# 加载MRC模型
mrc_model = MrcModel(mrc_config)
mrc_model.load_model()

# 加载classifier模型
classify_model = ClassifyModel(rel_config)
classify_model.load_model()

# 加载切分句子模型
doc_split_model = DocSplitModel(doc_split_config)
        
app = Sanic("cross-lingual-QA")

@app.route("/test", methods=['GET'])
async def test(request):
    res = mrc_model.predict('Panthers đã mất bao nhiêu điểm khi phòng thủ', '黑豹队的防守只丢了 308分，在联赛中排名第六，同时也以 24 次拦截领先国家橄榄球联盟 (NFL)，并且四次入选职业碗。职业碗防守截锋卡万?肖特以 11 分领先于全队，同时还有三次迫使掉球和两次重新接球。他的队友马里奥?爱迪生贡献了 6? 次擒杀。黑豹队的防线上有经验丰富的防守端锋贾里德?艾伦，他是五次>职业碗选手，曾以 136 次擒杀成为 NFL 职业生涯中的活跃领袖。另外还有在 9 场首发中就拿下 5 次擒杀的防>守端锋科尼?伊利。在他们身后，黑豹队的三名首发线卫中有两人入选了职业碗：托马斯?戴维斯和卢克?坎克利。戴维斯完成了 5? 次擒杀、四次迫使掉球和四次拦截，而坎克利带领球队在擒抱 (118) 中迫使两次掉球并拦截了他自己的四次传球。卡罗莱纳的第二防线有职业碗安全卫科特?科尔曼和职业碗角卫约什?诺曼，科尔曼带领球队>完成了职业生涯中高达七次拦截并同时贡献了 88 次擒抱，而诺曼在本赛季成长为一名封锁角卫并完成了四次拦>截，其中两次被判触地得分。')

    return json({'result': res,'return_code':'1'})

import time
@app.route("/passage_qa", methods=['POST'])
async def passage_qa(request):
    data = request.json
    params = data.keys()

    if 'question' in params and 'context' in params:
        question = data['question']
        context = data['context']
        res = mrc_model.predict(question, context)

        return json({'result': res,'return_code':'1'})

@app.route("/doc_qa", methods=['POST'])
async def doc_qa(request):
    data = request.json
    params = data.keys()

    if 'question' in params and 'context' in params:
        question = data['question']
        document = data['context']

        # 切分长文档
        passage = doc_split_model.split(document)
        logger.info(f"passage split result : {passage}")

        # 排序模型
        scores = classify_model.predict(question, passage)
        index_of_top_pred = np.argmax(scores)
        context = passage[index_of_top_pred]
        logger.info(f"passage selected by TOP1 : {context}")

        # MRC预测
        res = mrc_model.predict(question, context)

        return json({'result': res,'return_code':'1'})

app.run(host="0.0.0.0", port=8008)
