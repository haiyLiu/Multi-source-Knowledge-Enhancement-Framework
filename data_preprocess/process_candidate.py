import json
import os

import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from config import args
from base_methods import load_json_data, save_json_data, load_txt_data

def process_inverse(candidate_data):
    '''
    candidate_data: list[dict()]
    '''
    forward_candidate_data = list()
    backward_candidate_data = list()
    for item in candidate_data:
        if item["inverse"]:
            backward_candidate_data.append(item)
        else:
            forward_candidate_data.append(item)
    return forward_candidate_data, backward_candidate_data

def process_candidate(candidate_data, entity_dict, relation_data):
    re_candidate_data = list()
    for item in candidate_data:
        d = dict()
        inverse = item["inverse"]
        if not inverse:
            triplet = item["triplet"]
            head_id, relation, tail_id = triplet[0], triplet[1], triplet[2]
            head_name = entity_dict[head_id]
            tail_name = entity_dict[tail_id]
            trans_relation = relation_data[relation]

            d["head"] = head_name
            d["head_id"] = head_id
            d["relation"] = trans_relation
            d["tail"] = tail_name
            d["tail_id"] = tail_id
            
            d["forward_topk_ents"] = item["topk_names"]
            d["forward_ground_truth_rank"] = item["rank"]

            # 查找反向(也就是预测头实体)的条目
            inverse_triplet = [tail_id, relation, head_id]
            result = [candidate_item for candidate_item in candidate_data if candidate_item["triplet"]==inverse_triplet and candidate_item["inverse"]==True]
            for r_triplet in result:
                d["backward_topk_ents"] = r_triplet["topk_names"]
                d["backward_ground_truth_rank"] = r_triplet["rank"]

            re_candidate_data.append(d)
    return re_candidate_data

def list2dict(data, relation_data, triplet=False):
    '''
    data : list[dict()]
    将list转为dict，方便快速检索
    '''
    data_dict = dict()
    for item in data:
        if triplet:
            triplet = item["triplet"]
            head_id, relation, tail_id = triplet[0], relation_data[triplet[1]], triplet[2]
        else:
            head_id = item["head_id"]
            relation = item["relation"]
            tail_id = item["tail_id"]

        key = head_id + "||" + relation + "||" + tail_id
        if key in data_dict:
            print(key)
            # assert False
        data_dict[key] = item
    return data_dict

def merge_info_candidate(test_file_path, re_candidate_dict):
    '''
    test_file_path : str
    re_candidate_dict : dict()
    '''
    test_data = load_json_data(test_file_path)
    new_info = list()
    for item in test_data:
        try:
            del item["forward_answer"]
            del item["backward_answer"]
        except Exception:
            pass
        head_id = item["head_id"]
        relation = item["relation"]
        tail_id = item["tail_id"]
        key = head_id + "||" + relation + "||" + tail_id
        candidate_item = re_candidate_dict[key]

        item["forward_topk_ents"] = candidate_item["forward_topk_ents"]
        item["forward_ground_truth_rank"] = candidate_item["forward_ground_truth_rank"]
        item["backward_topk_ents"] = candidate_item["backward_topk_ents"]
        item["backward_ground_truth_rank"] = candidate_item["backward_ground_truth_rank"]
        new_info.append(item)

    return new_info

def add_candidate_name(entity_path, relation_path, candidate_path, save_path):
    '''
    处理SimKGC找出的候选实体，添加候选实体的name
    '''
    entity_data = load_json_data(entity_path)   # list()
    entity_dict = {item["entity_id"]: item["entity"] for item in entity_data}  # dict()
    relation_data = load_json_data(relation_path)   # dict()
    candidate_data = load_json_data(candidate_path) # list()

    re_candidate_data = process_candidate(candidate_data, entity_dict, relation_data)
    save_json_data(save_path, re_candidate_data)

def merge_method(relation_path, re_candidate_data_path, re_candidate_dict_path, test_dir):
    relation_data = load_json_data(relation_path)   # dict()
    re_candidate_data = load_json_data(re_candidate_data_path)
    re_candidate_dict = load_json_data(re_candidate_dict_path)
    test_files = [test_file for test_file in os.listdir(test_dir) if test_file.endswith("_info.json")]
    for test_file in test_files:
        test_file_path = os.path.join(test_dir, test_file)
        merge_info = merge_info_candidate(test_file_path, re_candidate_dict)

        save_file = test_file.split("_info.json")[0] + "_allinfo.json"
        save_path = os.path.join(test_dir, save_file)
        save_json_data(save_path, merge_info)

def process_train(path):
    '''
    path: train.txt
    '''
    entity_id2name = load_json_data(args.entity_id2name_path)
    relation_data = load_json_data(args.relation_path)
    train_data = load_txt_data(path)
    forward_train_data = dict()
    backward_train_data = dict()
    for item in train_data:
        h, r, t = item.split('\n')[0].split('\t')
        r = relation_data[r]
        if (h, r) not in forward_train_data:
            forward_train_data[(h, r)] = list()
        forward_train_data[(h, r)].append(entity_id2name[t])

        if (t, r) not in backward_train_data:
            backward_train_data[(t, r)] = list()
        backward_train_data[(t, r)].append(entity_id2name[h])

    return forward_train_data, backward_train_data


def remove_train_entity(data, file):
    '''
    针对WN18RR数据集，去除候选实体中包含train里面的实体
    data: list(dict())
    ''' 
    forward_train_data, backward_train_data = process_train(args.train_txt_path)
    new_list = list()
    for item in data:
        forward_candidates = item["forward_topk_ents"]
        backward_candidates = item["backward_topk_ents"]
        head = item["head"]
        tail = item["tail"]
        if args.task == "WN18RR":
            head = ' '.join(item["head"].split('_')[:-2])
            tail = ' '.join(item["tail"].split('_')[:-2])

        try:
            forward_filter_candidates = [m for m in forward_candidates if m not in forward_train_data[(item["head_id"], item["relation"])] or m == tail]
            backward_filter_candidates = [m for m in backward_candidates if m not in backward_train_data[(item["tail_id"], item["relation"])] or m == head]
            item["forward_filter_candidates"] = forward_filter_candidates
            item["backward_filter_candidates"] = backward_filter_candidates
            # print(file)
        except Exception:
            #可能存在一些三元组存在于test中但不在train中
            item["forward_filter_candidates"] = forward_candidates
            item["backward_filter_candidates"] = backward_candidates
        new_list.append(item)
    return new_list

if __name__ == "__main__":

    '''
    # 处理SimKGC找出的候选实体，添加候选实体的name
    entity_path = f"data/{args.task}/entities.json"
    relation_path = f"data/{args.task}/relations.json"
    candidate_path = f"data/{args.task}/candidate_results/test.json"

    # save_path = f"data/{args.task}/candidate_results/candidate_test.json"
    # add_candidate_name(entity_path, relation_path, candidate_path, save_path)
    # 处理SimKGC找出的候选实体，添加候选实体的name

    # 合并candidate实体和info里的信息
    re_candidate_data_path = f"data/{args.task}/candidate_results/candidate_test.json"
    re_candidate_dict_path = f"data/{args.task}/candidate_results/candidate_test_dict.json"

    # re_candidate_data = load_json_data(re_candidate_data_path)
    # relation_data = load_json_data(relation_path)
    # re_candidate_dict = list2dict(re_candidate_data, relation_data, triplet=False)
    # re_candidate_dict_path = f"data/{args.task}/candidate_results/candidate_test_dict.json"
    # save_json_data(re_candidate_dict_path, re_candidate_dict)
    # print(len(re_candidate_dict))
 
    test_dir = f"data/{args.task}/small_test"
    merge_method(relation_path, re_candidate_data_path, re_candidate_dict_path, test_dir)
    # 合并candidate实体和info里的信息
    '''

    # 针对数据集，去除候选实体中包含train里面的实体
    test_dir = args.test_dir
    # entity_name2id = load_json_data(args.entity_name2id_path)
    files = [file for file in os.listdir(test_dir) if file.endswith("allinfo.json")]
    for file in files:
        path = os.path.join(test_dir, file)
        data = load_json_data(path)
        new_data = remove_train_entity(data, file)
        save_json_data(path, new_data)


    
