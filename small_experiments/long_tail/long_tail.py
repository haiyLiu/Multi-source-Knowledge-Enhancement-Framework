import os
import re
import numpy as np
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, load_txt_data
from config import args


def split_entity_frequency(entity_frequency, max=10):
    result = {}
    for key, value in entity_frequency.items():
        if value < max:
            range_start, range_end = value, value
            range_key = f"{range_start}-{range_end}"
            if range_key not in result:
                result[range_key] = {}
            result[range_key][key] = value
    return result

def split_entity_frequency_bins(entity_frequency, max_split=100):
    result = {}
    max_value = max(entity_frequency.values())
    for key, value in entity_frequency.items():
        if value < max_split:  # 以max_split为界限
            # 按10为区间划分
            range_start = (value // 10) * 10
            range_end = range_start + 10
            range_key = f"{range_start}-{range_end}"    # 生成区间键，格式如 "0-10"
            if range_key not in result:
                result[range_key] = {}
            result[range_key][key] = value
        else:
            range_key = f"{max_split}-{max_value}"
            if range_key not in result:
                result[range_key] = {}
            result[range_key][key] = value
    return result

def load_answer(dir_path):
    llm_answer_dict = dict()
    try:
        for file in os.listdir(dir_path):
            path = os.path.join(dir_path, file)
            data = load_json_data(path)
            for item in data:
                #处理一下，将key变为h_id|r|t_id
                key = f"{item['head_id']}|{item['relation']}|{item['tail_id']}"
                llm_answer_dict[key] = item
    except Exception:
        data = load_json_data(dir_path)
        for item in data:
            #处理一下，将key变为h_id|r|t_id
            key = f"{item['head_id']}|{item['relation']}|{item['tail_id']}"
            llm_answer_dict[key] = item
    return llm_answer_dict

def load_split_data(data, entity_frequency):
    '''
    data: dict()
    entity_frequency: dict()
    根据实体频率数据entity_frequency，筛选出实体在test中的三元组，以及对应的answer
    '''
    new_test_forward = list()
    new_test_backward = list()
    for key, value in entity_frequency.items():
        for llm_key, llm_answer in data.items():
            parts = llm_key.split("|")
            #预测尾实体，那么尾实体应该是长尾
            if len(parts) > 2 and parts[2] == key:
                new_test_forward.append(llm_answer)
            #预测头实体，那么头实体应该是长尾
            if len(parts) > 2 and parts[0] == key:
                new_test_backward.append(llm_answer)
    return new_test_forward, new_test_backward

def process_llm_data(answer_path, entity_frequency, split_frequency_data, degree=False):
    llm_answer_data = load_answer(answer_path)

    for split_key, split_value in split_frequency_data.items():
        new_test_forward, new_test_backward = load_split_data(llm_answer_data, split_value)
        save_dir = os.path.join(args.long_tail_dir, args.eval_type)
        if degree:
            save_dir = os.path.join(args.long_tail_degree_dir, args.eval_type)
        save_json_data(os.path.join(save_dir, f"llm_forward_{split_key}.json"), new_test_forward)
        save_json_data(os.path.join(save_dir, f"llm_backward_{split_key}.json"), new_test_backward)

def process_transE_data(answer_path, entity_frequency, split_frequency_data, degree=False):
    transE_data = load_answer(answer_path)

    for split_key, split_value in split_frequency_data.items():
        new_test_forward, new_test_backward = load_split_data(transE_data, split_value)
        save_dir = os.path.join(args.long_tail_dir, args.eval_type)
        if degree:
            save_dir = os.path.join(args.long_tail_degree_dir, args.eval_type)
        save_json_data(os.path.join(save_dir, f"transE_forward_{split_key}.json"), new_test_forward)
        save_json_data(os.path.join(save_dir, f"transE_backward_{split_key}.json"), new_test_backward)

def process_simkgc_data(answer_path, entity_frequency, split_frequency_data, degree=False):
    simkgc_data = load_answer(answer_path)

    for split_key, split_value in split_frequency_data.items():
        new_test_forward, new_test_backward = load_split_data(simkgc_data, split_value)
        save_dir = os.path.join(args.long_tail_dir, args.eval_type)
        if degree:
            save_dir = os.path.join(args.long_tail_degree_dir, args.eval_type)
        save_json_data(os.path.join(save_dir, f"simkgc_forward_{split_key}.json"), new_test_forward)
        save_json_data(os.path.join(save_dir, f"simkgc_backward_{split_key}.json"), new_test_backward)


if __name__ == "__main__":
    entity_frequency = load_json_data(args.entity_frequency_path)
    
    if args.degree:
        split_frequency_data = split_entity_frequency(entity_frequency) # dict()  #degree=True
    else:
        split_frequency_data = split_entity_frequency_bins(entity_frequency) # dict()

    # LLM
    process_llm_data(os.path.join(args.new_rank_result_dir, args.llm_model), entity_frequency, split_frequency_data, degree=args.degree)

    # TransE
    # process_transE_data(os.path.join(args.transE_dir, "new_test.json"), entity_frequency, split_frequency_data, degree=args.degree)

    #SimKGC
    # process_simkgc_data(os.path.join(args.simkgc_dir, "new_test.json"), entity_frequency, split_frequency_data, degree=args.degree)