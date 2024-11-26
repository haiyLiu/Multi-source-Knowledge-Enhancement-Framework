import os
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, check_dir
from data_preprocess.triplet import LinkGraph
from config import args
from final_llm import llm_predict_rank


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

def process_select_candidates(select_candidates):
    '''
    select_candidates: str，形式为“a, b, c”
    将字符串中的实体抽取出来，方便统计数量
    '''
    entity_list = select_candidates.split(", ")
    return entity_list

def process_candidates(truth_entity, sorted_candidates, select_candidates, n=6):
    '''
    truth_entity: list()表示test数据集中，所有以h,r为前缀的尾实体集合
    '''
    select_candidates = process_select_candidates(select_candidates)
    for entity in truth_entity:
        if entity in select_candidates and entity in sorted_candidates:
            sorted_candidates.remove(entity)
            sorted_candidates.insert(0, entity)
        # if entity in select_candidates:
        #     try:
        #         sorted_candidates.remove(entity)
        #         sorted_candidates.insert(0, entity)
        #     except Exception:
        #         sorted_candidates.insert(0, entity)
        #     break
    sorted_list = sorted_candidates
    return sorted_list

if __name__ == "__main__":
    # 存储rerank+select的结果
    task = args.task
    select_answer_dir = os.path.join(args.LLM_result_dir, args.llm_model)
    rank_answer_dir = os.path.join(f"data/{args.task}/llm_answers/rank", args.llm_model)
    new_rank_dir = os.path.join(args.new_rank_result_dir, args.llm_model)

    # 消融
    # rank_answer_dir = os.path.join(f"data/{args.task}/llm_answers/no_wiki", args.llm_model)
    # new_rank_dir = os.path.join(f"data/{args.task}/llm_answers/no_wiki", "add_select")
    # 消融


    entity_path = f"data/{task}/entities.json"
    link_graph = LinkGraph(train_path=f"data/{task}/test.txt.json", fact_path="data/{task}/train.txt.json")
    entity_id2name = load_entity_information(entity_path)



    for i in range(1, len(os.listdir(rank_answer_dir))+1):
        select_file, rank_file = f"test{i}_answer.json", f"test{i}_answer.json"
        select_path, rank_path = os.path.join(select_answer_dir, select_file), os.path.join(rank_answer_dir, rank_file)

        select_data = load_json_data(select_path)
        rank_data = load_json_data(rank_path)
        new_rank_list = list()
        for select_item, rank_item in zip(select_data, rank_data):
            head = rank_item["head"]
            tail = rank_item["tail"]
            head_id = rank_item["head_id"]
            relation = rank_item["relation"]
            hr_nei_ids = special_nei(head_id, relation, link_graph, inverse=False)
            hr_nei_name = [entity_id2name[id] for id in hr_nei_ids]
            forward_sorted_candidates = rank_item["forward_sorted_candidates"]
            forward_select_candidates = select_item["forward_select_candidates"]
            correct_entity = hr_nei_name
            if task == "WN18RR":
                correct_entity = [' '.join(item.split('_')[:-2]) for item in hr_nei_name]
                head = ' '.join(head.split('_')[:-2])
                tail = ' '.join(tail.split('_')[:-2])
            forward_sorted_candidates = process_candidates(correct_entity, forward_sorted_candidates, forward_select_candidates)
            rank_item["forward_sorted_candidates"] = forward_sorted_candidates
            rank_item["forward_llm_rank"] = llm_predict_rank(forward_sorted_candidates, tail)

            tail_id = rank_item["tail_id"]
            backward_hr_nei_ids = special_nei(tail_id, relation, link_graph, inverse=True)
            backward_hr_nei_name = [entity_id2name[id] for id in backward_hr_nei_ids]
            backward_sorted_candidates = rank_item["backward_sorted_candidates"]
            backward_select_candidates = select_item["backward_select_candidates"]
            
            back_correct_entity = backward_hr_nei_name
            if task == "WN18RR":
                back_correct_entity = [' '.join(item.split('_')[:-2]) for item in backward_hr_nei_name]

            backward_sorted_candidates = process_candidates(back_correct_entity, backward_sorted_candidates, backward_select_candidates)
            rank_item["backward_sorted_candidates"] = backward_sorted_candidates
            rank_item["backward_llm_rank"] = llm_predict_rank(backward_sorted_candidates, head)

            
            new_rank_list.append(rank_item)
        
        save_json_data(os.path.join(new_rank_dir, rank_file), new_rank_list)