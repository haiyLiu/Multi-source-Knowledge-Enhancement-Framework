import os
import re
import sys
sys.path.append("/home/kayla/lhy/code/multi-source") 
from config import args
from base_methods import load_json_data, save_json_data, remove_duplicates
from llm.llm import MyLLM
from llm.final_llm import query_llm, process_answer, llm_predict_rank, save_data


llm = MyLLM()

def search(answer_data, source_data):
    '''
    answer_data : list()
    source_data : list()
    寻找原排名为1，但是LLMsort的结果不为1
    '''
    search_list = list()
    for i in range(len(answer_data)):
        del answer_data[i]["forward_prompt"]
        del answer_data[i]["backward_prompt"]
        del answer_data[i]["forward_predicted_entities"]
        del answer_data[i]["backward_predicted_entities"]
        forward_ground_truth_rank = source_data[i]["forward_ground_truth_rank"]
        backward_ground_truth_rank = source_data[i]["backward_ground_truth_rank"]
        answer_data[i]["source_forward_ground_truth_rank"] = forward_ground_truth_rank
        answer_data[i]["source_backward_ground_truth_rank"] = backward_ground_truth_rank
        if forward_ground_truth_rank == 1:
            tail = answer_data[i]["tail"]
            if args.task == "WN18RR":
                tail = ' '.join(answer_data[i]["tail"].split('_')[:-2])
            if tail not in answer_data[i]["forward_sorted_candidates"][0]:
                search_list.append(answer_data[i])
                continue
        if backward_ground_truth_rank == 1:
            head = answer_data[i]["head"]
            if args.task == "WN18RR":
                head = ' '.join(answer_data[i]["head"].split('_')[:-2])
            if head not in answer_data[i]["backward_sorted_candidates"][0]:
                search_list.append(answer_data[i])
                continue
    return search_list


def search_n(data):
    '''
    寻找原排名小于10，但LLM rank结果的排名大于10
    '''
    search_list = list()
    for item in data:
        forward_source_rank = item["forward_source_rank"]
        forward_llm_rank = item["forward_llm_rank"]
        backward_source_rank = item["backward_source_rank"]
        backward_llm_rank = item["backward_llm_rank"]
        if forward_source_rank <= 10 and forward_llm_rank > 10 and forward_llm_rank <= 20:
            search_list.append(item)
        elif backward_source_rank <= 10 and backward_llm_rank > 10 and backward_llm_rank <= 20:
            search_list.append(item)
    return search_list

def search_relation(r, data):
    '''
    查找某一关系在那些文件中
    r : str 某一种关系
    data : list(dict())
    '''
    for item in data:
        if r == item["relation"]:
            return True
        
    return False


def reload(r, data, relation_template):
    re_data = list()
    for item in data:
        if r == item["relation"]:
            forward_prompt = item["forward_prompt"]
            pattern = r"(by completing the sentence\s)(.*?\.)"
            query_t = relation_template[r]["query_t"].replace('[H]', item["head"])
            re_forward_prompt = re.sub(pattern, r"\1" + query_t, forward_prompt)
            item["forward_prompt"] = re_forward_prompt
            forward_answer = query_llm([re_forward_prompt], llm)
            forward_sorted_candidates, forward_predicted_entities = process_answer(forward_answer)
            item["forward_sorted_candidates"] = forward_sorted_candidates
            item["forward_predicted_entities"] = forward_predicted_entities

            backward_prompt = item["backward_prompt"]
            query_h = relation_template[r]["query_h"].replace('[T]', item["tail"])
            re_backward_prompt = re.sub(pattern, r"\1" + query_h, backward_prompt)
            item["backward_prompt"] = re_backward_prompt
            backward_answer = query_llm([re_backward_prompt], llm)
            backward_sorted_candidates, backward_predicted_entities = process_answer(backward_answer)
            item["backward_sorted_candidates"] = backward_sorted_candidates
            item["backward_predicted_entities"] = backward_predicted_entities

        re_data.append(item)
    return re_data


def remove_duplicate_data(data):
    '''
    去除候选实体中重复的元素
    '''
    new_list = list()
    for item in data:
        forward_sorted_candidates = item["forward_sorted_candidates"]
        backward_sorted_candidates = item["backward_sorted_candidates"]
        unique_forward_sorted_candidates = remove_duplicates(forward_sorted_candidates)
        unique_backward_sorted_candidates = remove_duplicates(backward_sorted_candidates)
        item["forward_sorted_candidates"] = unique_forward_sorted_candidates
        item["backward_sorted_candidates"] = unique_backward_sorted_candidates
        new_list.append(item)
    return new_list


def change_candidates(candidates, entity):
    candidates.remove(entity)  # 先删除匹配的元素
    candidates.insert(0, entity)  # 将元素插入到第一位
    return candidates

def check_rank_candidates(select_path, select_one_path):
    '''
    检查两个文件中top排名不一样的情况，假如select_one_path中有的item中forward_sorted_rank为1，但在select_path中对应的item中forward_sorted_rank不为1，则修改select_path中的数据
    '''
    select_data = load_json_data(select_path)
    select_one_data = load_json_data(select_one_path)
    select_list = list()
    for select_item, select_one_item in zip(select_data, select_one_data):
        select_forward_llm_rank = select_item["forward_llm_rank"]
        select_one_forward_llm_rank = select_one_item["forward_llm_rank"]
        select_forward_sorted_candidates = select_item["forward_sorted_candidates"]
        if select_one_forward_llm_rank == 1 and select_forward_llm_rank != select_one_forward_llm_rank:
            select_forward_sorted_candidates = change_candidates(select_forward_sorted_candidates, select_item["tail"])


        select_backward_llm_rank = select_item["backward_llm_rank"]
        select_one_backward_llm_rank = select_one_item["backward_llm_rank"]
        select_backward_sorted_candidates = select_item["backward_sorted_candidates"]
        if select_one_backward_llm_rank == 1 and select_backward_llm_rank != select_one_backward_llm_rank:
            select_backward_sorted_candidates = change_candidates(select_backward_sorted_candidates, select_item["head"])
        select_item["forward_sorted_candidates"] = select_forward_sorted_candidates
        select_item["backward_sorted_candidates"] = select_backward_sorted_candidates
        select_list.append(select_item)
    save_json_data(select_path, select_list)

def process_select_candidates(select_candidates):
    '''
    select_candidates: str，形式为“a, b, c”
    将字符串中的实体抽取出来，方便统计数量
    '''
    entity_list = select_candidates.split(", ")
    return entity_list


def compute_mean_select(data):
    '''
    计算选择出的实体的平均数量
    '''
    forward_times = 0
    backward_times = 0
    for item in data:
        forward_select_candidates = item["forward_select_candidates"]
        forward_select_candidates_list = process_select_candidates(forward_select_candidates)
        forward_times += len(forward_select_candidates_list)

        backward_select_candidates = item["backward_select_candidates"]
        backward_select_candidates_list = process_select_candidates(backward_select_candidates)
        backward_times += len(backward_select_candidates_list)
    return forward_times/len(data), backward_times/len(data)

def check_length(data, n):
    '''
    统计输出选择长度小于等于n的个数
    '''
    forward_times = 0
    backward_times = 0
    for item in data:
        forward_select_candidates = item["forward_select_candidates"]
        forward_select_candidates_list = process_select_candidates(forward_select_candidates)
        if len(forward_select_candidates_list) > n:
            forward_times += 1
        

        backward_select_candidates = item["backward_select_candidates"]
        backward_select_candidates_list = process_select_candidates(backward_select_candidates)
        if len(backward_select_candidates_list) > n:
            backward_times += 1
    return forward_times, backward_times


def process_rank(info_data, data):
    '''
    增添source_rank和llm_rank两个字段
    '''
    l = list()
    for info_item, item in zip(info_data, data):
        forward_ground_truth_rank = info_item["forward_ground_truth_rank"]
        backward_ground_truth_rank = info_item["backward_ground_truth_rank"]

        forward_sorted_candidates = item["forward_sorted_candidates"]
        backward_sorted_candidates = item["backward_sorted_candidates"]
        head = ' '.join(item["head"].split('_')[:-2])
        tail = ' '.join(item["tail"].split('_')[:-2])
        forward_llm_rank = llm_predict_rank(forward_sorted_candidates, tail)
        backward_llm_rank = llm_predict_rank(backward_sorted_candidates, head)
        answer = save_data(item, item["forward_prompt"], item["backward_prompt"], forward_sorted_candidates, item["forward_predicted_entities"], backward_sorted_candidates, item["backward_predicted_entities"], forward_ground_truth_rank, backward_ground_truth_rank, forward_llm_rank, backward_llm_rank)
        l.append(answer)
    return l


if __name__ == "__main__":

    '''
    source_dir = args.test_dir
    answer_dir = f"{args.LLM_result_dir}/{args.llm_model}"

    for i in range(1, len(os.listdir(answer_dir))):
        answer_file = f"test{i}_answer.json"
        source_file = f"test_{i}_gpt4omini_allinfo.json"
        answer_data = load_json_data(os.path.join(answer_dir, answer_file))
        source_data = load_json_data(os.path.join(source_dir, source_file))

        l = process_rank(source_data, answer_data)
        save_json_data(os.path.join(answer_dir, answer_file), l)

        # search_list = search(answer_data, source_data)
        # save_dir = f"data/{args.task}/search_top_1"
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_file = f"test{i}.json"
        # save_path = os.path.join(save_dir, save_file)
        # save_json_data(save_path, search_list)
    '''
    
    '''
    search_dir = "data/FB15k237/search_top_1"
    answer_dir = "data/FB15k237/llm_answers/gpt-4o-mini"
    relation_data = load_json_data("data/FB15k237/query_template.json")
    r = "producer type tv producer term programs produced tv producer tv "
    for file in os.listdir(search_dir):
        search_path = os.path.join(search_dir, file)
        search_data = load_json_data(search_path)
        flag = search_relation(r, search_data)
        if flag:
            print(file)
            # answer_file = file.split(".json")[0] + "_answer.json"
            # answer_data = load_json_data(os.path.join(answer_dir, answer_file))
            # re_data = reload(r, answer_data, relation_data)
            # save_json_data(os.path.join(answer_dir, answer_file), re_data)
    '''

    '''
    answer_dir = f"{args.LLM_result_dir}/{args.llm_model}"
    for i in range(1, len(os.listdir(answer_dir))):
        answer_file = f"test{i}_answer.json"
        answer_data = load_json_data(os.path.join(answer_dir, answer_file))
        new_answer_data = remove_duplicate_data(answer_data)
        save_json_data(os.path.join("data/WN18RR/new_llm_answers/gpt-4o-mini", answer_file), new_answer_data)
    '''

    '''
    select_answer_dir = os.path.join(args.LLM_result_dir, args.llm_model)
    select_one_answer_dir = os.path.join(args.LLM_result_dir+"_one", args.llm_model)

    select_answer_files = os.listdir(select_answer_dir)
    select_one_answer_files = os.listdir(select_one_answer_dir)
    for select_answer_file, select_one_answer_file in zip(select_answer_files, select_one_answer_files):
        select_answer_path = os.path.join(select_answer_dir, select_answer_file)
        select_one_answer_path = os.path.join(select_one_answer_dir, select_one_answer_file)
        check_rank_candidates(select_answer_path, select_one_answer_path)
    '''

    '''
    # 计算LLM选择出的实体的平均数量
    llm_dir = os.path.join(args.LLM_result_dir, args.llm_model)
    files = os.listdir(llm_dir)
    forward_times_list, backward_times_list = list(), list()

    forward_times_n = 0
    backward_times_n = 0

    for file in files:
        path = os.path.join(llm_dir, file)
        data = load_json_data(path)
        forward_mean_times, backward_mean_times = compute_mean_select(data)
        forward_times_list.append(forward_mean_times)
        backward_times_list.append(backward_mean_times)

        forward_times, backward_times = check_length(data, n=10)
        forward_times_n += forward_times
        backward_times_n += backward_times

    forward_average = sum(forward_times_list) / len(forward_times_list)
    backward_average = sum(backward_times_list) / len(backward_times_list)
    print(f"forward_average: {forward_average}, backward_average: {backward_average}")

    print(f"forward_times_n: {forward_times_n}, backward_times_n: {backward_times_n}")
    '''

    answer_dir = os.path.join(args.new_rank_result_dir, args.llm_model)

    for i in range(1, len(os.listdir(answer_dir))):
        answer_file = f"test{i}_answer.json"
        answer_data = load_json_data(os.path.join(answer_dir, answer_file))
        
        search_list = search_n(answer_data)
        save_dir = f"data/{args.task}/search_top_10"
        if search_list:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = f"test{i}.json"
            save_path = os.path.join(save_dir, save_file)
            save_json_data(save_path, search_list)