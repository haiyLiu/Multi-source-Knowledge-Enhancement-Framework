import os
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, check_dir
from data_preprocess.triplet import LinkGraph
from config import args


def load_entity_information(path):
    entity_list = load_json_data(path)
    entity_id2name = dict()
    for item in entity_list:
        id = item["entity_id"]
        name = item["entity"]
        entity_id2name[id] = name
    return entity_id2name


def special_nei(entity_id, relation, link_graph, inverse):
    relation_neighbor_ids = link_graph.get_special_relation_neighbor_ids(entity_id, relation, inverse)
    return relation_neighbor_ids

def calculate_hits_k(correct_entity, sorted_candidates, k=1):
    '''
    计算hits@k指标
    :param correct_entity: list()  正确的实体
    :param sorted_candidates: list()    排序后的候选答案列表
    :param k: int   截止排名位置K
    :return: 如果正确实体出现在前K个候选答案中，返回1；否则返回0
    '''
    # 确保K不超过候选答案数量
    k = min(k, len(sorted_candidates))
    # 检查正确实体是否在前K个候选答案中
    hits = any(candidate in correct_entity for candidate in sorted_candidates[:k])
    return 1 if hits else 0

def calculate_mrr(correct_entity, sorted_candidates):
    '''
    计算MRR指标
    correct_entity: list()
    sorted_candidates: list()
    '''
    reciprocal_ranks = []
    
    # 遍历每个正确实体
    for entity in correct_entity:
        try:
            # 找到正确实体在候选列表中的索引（排名从0开始，因此加1）
            rank = sorted_candidates.index(entity) + 1
            # 计算排名的倒数
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            # 如果实体不在候选列表中，倒数为0
            reciprocal_ranks.append(0)
    
    # 计算 MRR，考虑到可能有实体不在候选列表中
    mrr = max(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr

if __name__ == "__main__":
    task = args.task
    answer_dir = os.path.join(args.new_rank_result_dir, args.llm_model)

    # 消融
    # answer_dir = f"data/{task}/llm_answers/no_wiki/add_select"
    # 消融

    entity_path = f"data/{task}/entities.json"
    link_graph = LinkGraph(train_path=f"data/{task}/test.txt.json", fact_path="data/{task}/train.txt.json")
    entity_id2name = load_entity_information(entity_path)

    forward_hits_1_sum = 0
    forward_hits_3_sum = 0
    forward_hits_10_sum = 0
    forward_mrr_sum = 0

    backward_hits_1_sum = 0
    backward_hits_3_sum = 0
    backward_hits_10_sum = 0
    backward_mrr_sum = 0
    times = 0
    for file in os.listdir(answer_dir):
        answer_data = load_json_data(os.path.join(answer_dir, file))
        for item in answer_data:
            head_id = item["head_id"]
            relation = item["relation"]
            hr_nei_ids = special_nei(head_id, relation, link_graph, inverse=False)
            hr_nei_name = [entity_id2name[id] for id in hr_nei_ids]
            forward_sorted_candidates = item["forward_sorted_candidates"]
            correct_entity = hr_nei_name
            if task == "WN18RR":
                correct_entity = [' '.join(item.split('_')[:-2]) for item in hr_nei_name]
            
            forward_hits_1_sum += calculate_hits_k(correct_entity, forward_sorted_candidates)
            forward_hits_3_sum += calculate_hits_k(correct_entity, forward_sorted_candidates, k=3)
            forward_hits_10_sum += calculate_hits_k(correct_entity, forward_sorted_candidates, k=10)
            forward_mrr_sum += calculate_mrr(correct_entity, forward_sorted_candidates)

            tail_id = item["tail_id"]
            backward_hr_nei_ids = special_nei(tail_id, relation, link_graph, inverse=True)
            backward_hr_nei_name = [entity_id2name[id] for id in backward_hr_nei_ids]
            backward_sorted_candidates = item["backward_sorted_candidates"]
            
            back_correct_entity = backward_hr_nei_name
            if task == "WN18RR":
                back_correct_entity = [' '.join(item.split('_')[:-2]) for item in backward_hr_nei_name]
            backward_hits_1_sum += calculate_hits_k(back_correct_entity, backward_sorted_candidates)
            backward_hits_3_sum += calculate_hits_k(back_correct_entity, backward_sorted_candidates, k=3)
            backward_hits_10_sum += calculate_hits_k(back_correct_entity, backward_sorted_candidates, k=10)
            backward_mrr_sum += calculate_mrr(back_correct_entity, backward_sorted_candidates)
            
        times += len(answer_data)
    print(f"forward_hits_1: {forward_hits_1_sum/times}, forward_hits_3: {forward_hits_3_sum/times}, forward_hits_10: {forward_hits_10_sum/times}")
    print(f"backward_hits_1: {backward_hits_1_sum/times}, backward_hits_3: {backward_hits_3_sum/times}, backward_hits_10: {backward_hits_10_sum/times}")
    print(f"forward_mrr: {forward_mrr_sum/times}, backward_mrr: {backward_mrr_sum/times}")

    mean_hits_1_sum = (forward_hits_1_sum + backward_hits_1_sum) / 2
    mean_hits_3_sum = (forward_hits_3_sum + backward_hits_3_sum) / 2
    mean_hits_10_sum = (forward_hits_10_sum + backward_hits_10_sum) / 2
    mean_mrr_sum = (forward_mrr_sum + backward_mrr_sum) / 2

    print(f"mean_hits_1: {mean_hits_1_sum/times}, mean_hits_3: {mean_hits_3_sum/times}, mean_hits_10: {mean_hits_10_sum/times}, mean_mrr: {mean_mrr_sum/times}")