import os
import re
import numpy as np
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, load_txt_data
from config import args
from data_preprocess.process_candidate import process_inverse


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
            
            d["forward_source_rank"] = item["rank"]
            d["forward_sorted_candidates"] = item["topk_names"]

            # 查找反向(也就是预测头实体)的条目
            inverse_triplet = [tail_id, relation, head_id]
            result = [candidate_item for candidate_item in candidate_data if candidate_item["triplet"]==inverse_triplet and candidate_item["inverse"]==True]
            for r_triplet in result:
                d["backward_source_rank"] = r_triplet["rank"]
                d["backward_sorted_candidates"] = r_triplet["topk_names"]

            re_candidate_data.append(d)
    return re_candidate_data


def process_formart_data(entity_path, relation_path, data_path):
    '''
    处理SimKGC和TransE找出的候选实体，添加候选实体的name
    '''
    entity_id2name = load_json_data(entity_path)  # dict()
    relation_data = load_json_data(relation_path)   # dict()
    candidate_data = load_json_data(data_path) # list()

    format_data = process_candidate(candidate_data, entity_id2name, relation_data)
    return format_data
    

if __name__ == "__main__":

    # transE_test_path = os.path.join(args.transE_dir, "test.json")
    # transE_data = process_formart_data(args.entity_id2name_path, args.relation_path, transE_test_path)
    # save_path = os.path.join(args.transE_dir, "new_test.json")
    # save_json_data(save_path, transE_data)

    simkgc_test_path = os.path.join(args.simkgc_dir, "test.json")
    simkgc_data = process_formart_data(args.entity_id2name_path, args.relation_path, simkgc_test_path)
    save_path = os.path.join(args.simkgc_dir, "new_test.json")
    save_json_data(save_path, simkgc_data)