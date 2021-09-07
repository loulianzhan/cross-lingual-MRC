#!/usr/bin/env python
# coding=utf-8


"""
convert raw input into example
raw input : [{"question" : xxx , "context" : yyy}]
output : exmaple
"""

from datasets import Dataset

def convert_into_dataset_instance(question, context, question_column_name, context_column_name, id_name):
    if isinstance(question, str) and isinstance(context, str):
        example_id = 0
        input_dict = {id_name: [example_id], question_column_name : [question], context_column_name : [context]}
        input_ds = Dataset.from_dict(input_dict)
    elif isinstance(question, str) and isinstance(context, list):
        input_dict = {id_name: [], question_column_name : [], context_column_name : []}
        for i , cont in enumerate(context):
            input_dict[id_name].append(i)
            input_dict[question_column_name].append(question)
            input_dict[context_column_name].append(cont)
        input_ds = Dataset.from_dict(input_dict)
    elif isinstance(question, list) and isinstance(context, str):
        input_dict = {id_name: [], question_column_name : [], context_column_name : []}
        for i , ques in enumerate(question):
            input_dict[id_name].append(i)
            input_dict[question_column_name].append(ques)
            input_dict[context_column_name].append(context)
        input_ds = Dataset.from_dict(input_dict)
    elif isinstance(question, list) and isinstance(context, list) and len(question) == len(context):
        input_dict = {id_name: [], question_column_name : [], context_column_name : []}
        for i , cont in enumerate(context):
            input_dict[id_name].append(i)
            input_dict[question_column_name].append(question[i])
            input_dict[context_column_name].append(cont)
        input_ds = Dataset.from_dict(input_dict)
    else:
        logger.error("illeagal input data, correct input for mrc prediction must have 2 arguments: question , context")
    return input_ds
