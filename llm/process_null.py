import os
import re
import json

from llm import MyLLM
from final_llm import load_infomation, query_llm, process_answer
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, check_dir
from config import args

def check_length_candidates(item):
    '''
    判断LLM输出的候选实体的长度，如果不等于20需要修改
    '''
    forward_sorted_candidates = item["forward_sorted_candidates"]
    backward_sorted_candidates = item["backward_sorted_candidates"]
    if len(forward_sorted_candidates) == 0 or len(backward_sorted_candidates) == 0: #
        return item
    return None


def process(item):
    '''
    处理llm输出的候选和预测实体为[]或LLM回答的候选实体排序列表！=20（即存在未回答全部候选实体的情况）的情况
    item : dict()
    '''
    if len(item["forward_sorted_candidates"]) == 0 or len(item["forward_predicted_entities"]) == 0 or len(item["forward_sorted_candidates"]) != 20:
        forward_prompt = item["forward_prompt"]
        forward_answer = query_llm(forward_prompt, llm, model=args.llm_model)
        forward_sorted_candidates, forward_predicted_entities = process_answer(forward_answer)
        if len(forward_sorted_candidates) == 0 or len(forward_sorted_candidates) == 0 or len(item["forward_sorted_candidates"]) != 20:
            forward_answer = query_llm(forward_prompt, llm, model=args.llm_model)
            forward_sorted_candidates, forward_predicted_entities = process_answer(forward_answer)
        item["forward_sorted_candidates"] = forward_sorted_candidates
        item["forward_predicted_entities"] = forward_predicted_entities

    if len(item["backward_sorted_candidates"]) == 0 or len(item["backward_predicted_entities"]) == 0 or len(item["backward_sorted_candidates"]) != 20:
        backward_prompt = item["backward_prompt"]
        backward_answer = query_llm(backward_prompt, llm, model=args.llm_model)
        backward_sorted_candidates, backward_predicted_entities = process_answer(backward_answer)
        if len(backward_sorted_candidates) == 0 or len(backward_sorted_candidates) == 0 or len(item["backward_sorted_candidates"]) != 20:
            backward_answer = query_llm(backward_prompt, llm, model=args.llm_model)
            backward_sorted_candidates, backward_predicted_entities = process_answer(backward_answer)
        item["backward_sorted_candidates"] = backward_sorted_candidates
        item["backward_predicted_entities"] = backward_predicted_entities
    return item

if __name__ == "__main__":
    llm = MyLLM()
    dir_path = os.path.join(args.LLM_result_dir, args.llm_model)
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        data = load_json_data(path)
        answer_list = list()
        error_list = list()
        for item in data:
            # answer_list.append(process(item))
            error_list.append(check_length_candidates(item))
        # save_json_data(path, answer_list)
        if all(element is None for element in error_list):
            continue
        save_dir = f"data/{args.task}/answer_null"
        file_name = file.split(".json")[0]
        save_file = f"{file_name}_error.json"
        save_json_data(os.path.join(save_dir, save_file), error_list)