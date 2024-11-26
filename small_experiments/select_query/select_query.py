import numpy as np
import os
import random
import re
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, check_dir
from config import args
from small_experiments.long_tail.long_tail import split_entity_frequency_bins


def calculate_proportion(data_dict):
    '''
    data_dict: dict(dict())
    '''
    # 计算data_dict1中各个key所占的百分比
    # all_frequency = 0
    each_frequency = dict()
    for range, values in data_dict.items():
        each_frequency[range] = len(values)

    # 计算总和
    total = sum(each_frequency.values())
    # 计算每一项的百分比
    each_percentages = {key: (value / total) for key, value in each_frequency.items()}

    return each_percentages


def calculate_each_quantity(percentages, all_quantity=1000):
    '''
    根据百分比计算每一个group要采样多少项
    '''
    sample_counts = {key: round(value * all_quantity) for key, value in percentages.items()}
    return sample_counts

def random_select(forward_data, backward_data, counts):
    '''
    forward_data: list()
    backward_data: list()
    counts: int
    '''
    forward_counts = counts // 2
    backward_counts = counts - forward_counts
    forward_select = random.sample(forward_data, forward_counts)
    backward_select = random.sample(backward_data, backward_counts)
    return forward_select, backward_select

if __name__ == "__main__":
    data_path = os.path.join(args.long_tail_dir, args.eval_type)
    save_dir = os.path.join(args.select_query_dir, args.eval_type)

    entity_frequency = load_json_data(args.entity_frequency_path)
    split_frequency_data = split_entity_frequency_bins(entity_frequency) # dict()

    split_percentages_data = calculate_proportion(split_frequency_data)
    sample_counts = calculate_each_quantity(split_percentages_data, all_quantity=200)

    for range, counts in sample_counts.items():
        forward_file = f"{args.eval_type}_forward_{range}.json"
        backward_file = f"{args.eval_type}_backward_{range}.json"
        forward_path = os.path.join(data_path, forward_file)
        backward_path = os.path.join(data_path, backward_file)
        forward_data = load_json_data(forward_path)
        backward_data = load_json_data(backward_path)

        forward_select, backward_select = random_select(forward_data, backward_data, counts)
        save_json_data(os.path.join(save_dir, forward_file),forward_select)
        save_json_data(os.path.join(save_dir, backward_file),backward_select)


    
