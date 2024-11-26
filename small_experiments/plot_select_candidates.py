import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/home/kayla/lhy/code/multi-source") 
from config import args
from base_methods import load_json_data, save_json_data, remove_duplicates
from llm.llm import MyLLM
from llm.final_llm import query_llm, process_answer
from data_preprocess.process_llmanswer import process_select_candidates


if __name__ == "__main__":
    task = args.task
    answer_dir = os.path.join(args.LLM_result_dir, args.llm_model)
    # entity_path = f"data/{task}/entities.json"
    # link_graph = LinkGraph(train_path=f"data/{task}/test.txt.json", fact_path="data/{task}/train.txt.json")
    # entity_id2name = load_entity_information(entity_path)


    forward_len_dict = dict()
    backward_len_dict = dict()
    for file in os.listdir(answer_dir):
        answer_data = load_json_data(os.path.join(answer_dir, file))
        for item in answer_data:
            forward_select_candidates = process_select_candidates(item["forward_select_candidates"])
            forward_select_candidates_length = len(forward_select_candidates)
            if forward_select_candidates_length not in forward_len_dict:
                forward_len_dict[forward_select_candidates_length] = 0
            forward_len_dict[forward_select_candidates_length] += 1


            backward_select_candidates = process_select_candidates(item["backward_select_candidates"])
            backward_select_candidates_length = len(backward_select_candidates)
            if backward_select_candidates_length not in backward_len_dict:
                backward_len_dict[backward_select_candidates_length] = 0
            backward_len_dict[backward_select_candidates_length] += 1

    # 计算dict1中各key所占的百分比
    total1 = sum(forward_len_dict.values())
    forward_len_dict = {k: (v / total1) for k, v in forward_len_dict.items()}

    # 计算dict2中各key所占的百分比
    total2 = sum(backward_len_dict.values())
    backward_len_dict = {k: (v / total2) for k, v in backward_len_dict.items()}
    
    # 绘制曲线图
    plt.figure(figsize=(12, 8))
    # 绘制第一个直方图（dict1的值）
    plt.subplot(1, 2, 1)
    plt.bar(forward_len_dict.keys(), forward_len_dict.values(), color='skyblue')
    plt.xlabel('Key')
    plt.ylabel('Value')
    plt.title('Histogram of forward_len')


    # 绘制第二个直方图（dict2的值）
    plt.subplot(1, 2, 2)
    plt.bar(backward_len_dict.keys(), backward_len_dict.values(), color='salmon')
    plt.xlabel('Key')
    plt.ylabel('Value')
    plt.title('Histogram of backward_len')

    plt.tight_layout()
    plt.show()

    plt.savefig("data/WN18RR/images/image_select_len_percetage.jpg")


