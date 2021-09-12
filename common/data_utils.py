#!/usr/bin/env python
# coding=utf-8


"""
convert raw input into example
raw input : [{"question" : xxx , "context" : yyy}]
output : exmaple
"""

from datasets import Dataset
import numpy as np

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

# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat
