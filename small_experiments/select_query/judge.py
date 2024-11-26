# 使用gpt4o评估随机抽取的1000个query的合理性

import os
import re
import sys 
import openai
from collections import defaultdict
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data
from config import args
from llm.llm import MyLLM

def process_entity_desc(entity_desc_path):
    '''
    处理entity_desc，使其变为id2desc，方便后续查找
    entity_desc_path
    '''
    entity_desc = load_json_data(entity_desc_path)  #list()
    entity_desc_dict = dict()
    for item in entity_desc:
        entity_id = item["entity_id"]
        entity_desc = item["entity_desc"]
        entity_desc_dict[entity_id] = entity_desc
    return entity_desc_dict

def construct_prompt(item, entity_id2desc, relation, relation_template, entity_name2id, reverse=False):
    '''
    构造prompt
    item: dict()
    '''
    head_id, tail_id = item["head_id"], item["tail_id"]
    forward_sorted_candidates = item["forward_sorted_candidates"]
    backward_sorted_candidates = item["backward_sorted_candidates"]
    if args.task == "WN18RR":
        head = ' '.join(item["head"].split('_')[:-2])
        tail = ' '.join(item["tail"].split('_')[:-2])
    else:
        head, tail = item["head"], item["tail"]
    sorted_tail = forward_sorted_candidates[0]
    sorted_head = backward_sorted_candidates[0]
    r_template = relation_template[relation[item["relation"]]]

    knowledge_query = ""
    if reverse:
        # backward
        sentence = r_template.replace('[X]', sorted_head).replace('[Y]', tail).strip()
        tail_desc = entity_id2desc[tail_id]
        head_desc = ""
        if sorted_head in entity_name2id:
            head_desc = entity_id2desc[entity_name2id[sorted_head]]
        # backward
        knowledge_query = f"Background Knowledge: <{sorted_head}: {head_desc}; {tail}: {tail_desc}>"
    else:
        # forward
        sentence = r_template.replace('[X]', head).replace('[Y]', sorted_tail).strip()
        head_desc = entity_id2desc[head_id]
        tail_desc = ""
        if sorted_tail in entity_name2id:
            tail_desc = entity_id2desc[entity_name2id[sorted_tail]]
        
        knowledge_query = f"Background Knowledge: <{head}: {head_desc}; {sorted_tail}: {tail_desc}>"
        # forward
    
    

    task_query = "Please act as an impartial judge and evaluate the correctness of the sentence provided by users. Your evaluation should consider correctness and helpfulness. Please avoid any bias and evaluate as objectively as possible. Do not allow the length of your responses to influence your evaluation."
    
    sentence_query = f"Sentence: <{sentence}>"
    query = f"Your work is to evaluate the correctness of the sentence based on the background knowledge provided and your own knowledge. The judgment results of the sentence are divided into three categories: Correct, Wrong, and Uncertain. Correct means that the sentence is correct, Uncertain means that it is impossible to judge whether the sentence is correct or not, and Wrong means that the sentence is incorrect. After providing your explanation, output your final verdict following this format: <The sentence was judges as [Correct or Uncertain or Wrong]>'"

    conversation = [
        {"role": "system", "content": task_query},
        {"role": "assistant", "content": knowledge_query},
        {"role": "assistant", "content": sentence_query},
        {"role": "user", "content": query}
    ]
    prompt = f"{task_query} \n {knowledge_query} \n {sentence_query} \n {query}"


    return conversation, prompt, sentence


def construct_prompt_v2(item, entity_id2desc, relation, relation_template, entity_name2id, reverse=False):
    '''
    构造prompt,相较于construct_prompt，更新了prompt的说法
    item: dict()
    '''
    head_id, tail_id = item["head_id"], item["tail_id"]
    forward_sorted_candidates = item["forward_sorted_candidates"]
    backward_sorted_candidates = item["backward_sorted_candidates"]
    head, tail = item["head"], item["tail"]
    sorted_tail = forward_sorted_candidates[0]
    sorted_head = backward_sorted_candidates[0]
    r_template = relation_template[relation[item["relation"]]]

    knowledge_query = ""
    if reverse:
        # backward
        sentence = r_template.replace('[X]', sorted_head).replace('[Y]', tail).strip()
        tail_desc = entity_id2desc[tail_id]
        head_desc = ""
        if sorted_head in entity_name2id:
            head_desc = entity_id2desc[entity_name2id[sorted_head]]
        # backward
        knowledge_query = f"Background Knowledge: <{sorted_head}: {head_desc}; {tail}: {tail_desc}>"
    else:
        # forward
        sentence = r_template.replace('[X]', head).replace('[Y]', sorted_tail).strip()
        head_desc = entity_id2desc[head_id]
        tail_desc = ""
        if sorted_tail in entity_name2id:
            tail_desc = entity_id2desc[entity_name2id[sorted_tail]]
        
        knowledge_query = f"Background Knowledge: <{head}: {head_desc}; {sorted_tail}: {tail_desc}>"
        # forward
    
    

    task_query = "Please act as an impartial judge and evaluate the correctness of the sentence provided by users. Your evaluation should consider correctness and helpfulness. Please avoid any bias and evaluate as objectively as possible. Do not allow the length of your responses to influence your evaluation."
    
    notice_query = f"Please note two points: (1)The knowledge provided you can be used as a reference, but it CANNOT be used as a criterion for judging the correctness of the sentence. (2)Note that the judgment results of the sentence are divided into two categories: Correct, and Wrong. Correct means that the sentence is correct, and Wrong means that the sentence is incorrect."
    query = f"Sentence: <{sentence}>. Your work is to evaluate the correctness of the sentence step by step based on your brainstormed knowledge of the sentence.  After providing your explanation, output your final verdict following this format: <The sentence was judges as [Correct or Wrong]>"

    conversation = [
        {"role": "system", "content": task_query},
        {"role": "assistant", "content": knowledge_query},
        {"role": "assistant", "content": notice_query},
        {"role": "user", "content": query}
    ]
    prompt = f"{task_query} \n {knowledge_query} \n {notice_query} \n {query}"


    return conversation, prompt, sentence


def construct_prompt_v3(item, entity_id2desc, relation, relation_template, entity_name2id, reverse=False):
    '''
    构造prompt，相较于construct_prompt_v2，去除了实体描述，增加了wiki的知识
    item: dict()
    '''
    head_id, tail_id = item["head_id"], item["tail_id"]
    forward_sorted_candidates = item["forward_sorted_candidates"]
    backward_sorted_candidates = item["backward_sorted_candidates"]
    forward_prompt = item["forward_prompt"]
    backward_prompt = item["backward_prompt"]
    if args.task == "WN18RR":
        head = ' '.join(item["head"].split('_')[:-2])
        tail = ' '.join(item["tail"].split('_')[:-2])
    else:
        head, tail = item["head"], item["tail"]
    sorted_tail = forward_sorted_candidates[0]
    sorted_head = backward_sorted_candidates[0]
    r_template = relation_template[relation[item["relation"]]]
    pattern = r'\*\*Knowledge from Wiki knowledge base\*\*:\s*(\[[^\]]+\])'
    forward_wiki = re.search(pattern, forward_prompt).group(1)
    backward_wiki = re.search(pattern, backward_prompt).group(1)

    knowledge_query = ""
    if reverse:
        # backward
        sentence = r_template.replace('[X]', sorted_head).replace('[Y]', tail).strip()
        # tail_desc = entity_id2desc[tail_id]
        # head_desc = ""
        # if sorted_head in entity_name2id:
        #     head_desc = entity_id2desc[entity_name2id[sorted_head]]
        # backward
        knowledge_query = f"Background Knowledge: {backward_wiki}"
    else:
        # forward
        sentence = r_template.replace('[X]', head).replace('[Y]', sorted_tail).strip()
        # head_desc = entity_id2desc[head_id]
        # tail_desc = ""
        # if sorted_tail in entity_name2id:
        #     tail_desc = entity_id2desc[entity_name2id[sorted_tail]]
        
        knowledge_query = f"Background Knowledge: {forward_wiki}"
        # forward
    
    

    task_query = "Please act as an impartial judge and evaluate the correctness of the sentence provided by users. Your evaluation should consider correctness and helpfulness. Please avoid any bias and evaluate as objectively as possible. Do not allow the length of your responses to influence your evaluation."
    
    notice_query = f"Please note two points: (1)The knowledge provided you can be used as a reference, but it CANNOT be used as a criterion for judging the correctness of the sentence. (2)Note that the judgment results of the sentence are divided into two categories: Correct, and Wrong. Correct means that the sentence is correct, and Wrong means that the sentence is incorrect."
    query = f"Sentence: <{sentence}>. Your work is to evaluate the correctness of the sentence step by step based on your brainstormed knowledge of the sentence.  After providing your explanation, output your final verdict following this format: <The sentence was judges as [Correct or Wrong]>"

    conversation = [
        {"role": "system", "content": task_query},
        {"role": "assistant", "content": knowledge_query},
        {"role": "assistant", "content": notice_query},
        {"role": "user", "content": query}
    ]
    prompt = f"{task_query} \n {knowledge_query} \n {notice_query} \n {query}"


    return conversation, prompt, sentence



def query_llm(conversation, llm, model="gpt-4o"):
    '''
    conversation : list()
    '''
    response = llm.client.chat.completions.create(
        model=model,  #选择的GPT模型名称
        messages=conversation,
        temperature = 0,
        presence_penalty = 0,
        frequency_penalty = 0,
        top_p = 1
        # temperature=0.7,
    )
    answer = response.choices[0].message.content
    return answer


def process_answer(answer):
    # 使用正则表达式提取指定句子
    sentence_match = re.search(r"The sentence was judged as (\w+)", answer)
    if sentence_match:
        extracted_sentence = sentence_match.group(0)  # 提取完整句子
        extracted_keyword = sentence_match.group(1)  # 提取关键字
        return extracted_keyword
    else:
        return answer
    
def save_new_item(item, prompt, response, answer, sentence):
    '''
    构造关于判断的item
    '''
    new_item = dict()
    new_item["head_id"], new_item["head"], new_item["relation"], new_item["tail_id"], new_item["tail"] = item["head_id"], item["head"], item["relation"], item["tail_id"], item["tail"]
    new_item["sentence"] = sentence
    new_item["answer"] = answer
    new_item["response"] = response
    new_item["prompt"] = prompt

    return new_item


def judge():
    llm = MyLLM()
    select_dir = os.path.join(args.select_query_dir, args.eval_type)
    save_dir = os.path.join(args.evaluation_dir, args.eval_type, args.eval_llm_model)
    entity_desc_path = args.entity_desc_path
    entity_id2desc = process_entity_desc(entity_desc_path)
    entity_name2id = load_json_data(args.entity_name2id_path)
    relation = load_json_data(args.relation_path)
    relation = {v:k for k,v in relation.items()}
    relation_template = load_json_data(args.relation_template_path)
    # files = ["llm_backward_0-10.json", "llm_backward_10-20.json", "llm_backward_20-30.json", "llm_backward_30-40.json", "llm_backward_40-50.json"]
    # files = ["llm_backward_50-60.json", "llm_backward_60-70.json", "llm_backward_70-80.json", "llm_backward_80-90.json", "llm_backward_90-100.json", "llm_backward_100-7614.json"]

    # files = ["llm_forward_0-10.json", "llm_forward_10-20.json", "llm_forward_20-30.json"]
    # files = ["llm_forward_30-40.json", "llm_forward_40-50.json", "llm_forward_50-60.json", "llm_forward_60-70.json", ]
    files = ["llm_forward_70-80.json", "llm_forward_80-90.json", "llm_forward_90-100.json", "llm_forward_100-7614.json"]
    


    for file in files:
        data = load_json_data(os.path.join(select_dir, file))
        new_list = list()
        for item in data:
            if "forward" in file:
                conversation, prompt, sentence = construct_prompt_v3(item, entity_id2desc, relation, relation_template, entity_name2id)
            else:
                conversation, prompt, sentence = construct_prompt_v3(item, entity_id2desc, relation, relation_template, entity_name2id, reverse=True)
            response = query_llm(conversation, llm, args.eval_llm_model)
            answer = process_answer(response)

            new_item = save_new_item(item, prompt, response, answer, sentence)

            new_list.append(new_item)
        save_json_data(os.path.join(save_dir, file), new_list)

def calculate():
    judge_dir = os.path.join(args.evaluation_dir, args.eval_type, args.eval_llm_model)
    files = os.listdir(judge_dir)
    correct, uncertain, wrong, other= 0, 0, 0, 0
    times = 0
    
    for file in files:
        other_list = list()
        path = os.path.join(judge_dir, file)
        data = load_json_data(path)
        for item in data:
            answer = item["answer"]
            if answer.lower().strip() == "correct":
                correct += 1
            elif answer.lower().strip() == "uncertain":
                uncertain += 1
            elif answer.lower().strip() == "wrong":
                wrong += 1
            else:
                other += 1
                other_list.append(item)
        times += len(data)
        if other_list:
            save_json_data(os.path.join(judge_dir, "other", file), other_list)
    print(f"correct: {correct}, uncertain: {uncertain}, wrong: {wrong}, other: {other}, times: {times}")
    print(f"correct percetage: {correct/times}, uncertain percetage: {uncertain/times}, wrong percetage: {wrong/times}, other percetage: {other/times}")


def process_other():
    judge_dir = os.path.join(args.evaluation_dir, args.eval_type, args.eval_llm_model)
    # other_dir = os.path.join(judge_dir, "other")
    files = os.listdir(judge_dir)
    for file in files:
        new_list = list()
        judge_data = load_json_data(os.path.join(judge_dir, file))
        for item in judge_data:
            answer = item["answer"]
            answer = process_answer(answer)
            new_item = save_new_item(item, item["prompt"], item["response"], answer, item["sentence"])
            new_list.append(new_item)
        save_json_data(os.path.join(judge_dir, file), new_list)



def process_merge(merged_item):
    '''
    merged_item: dict()
    将其处理为list形式
    '''
    new_list = list()
    for key,value in merged_item.items():
        new_list.append(value)
    return new_list


def merge_result(wiki=False):
    '''
    合并gpt4o，gpt4o-mini与自动化评估的结果
    '''
    evaluation_dir = os.path.join(args.evaluation_dir, args.eval_type)
    save_dir = os.path.join(args.evaluation_dir, args.eval_type, "merge")
    # 获取目录下的文件夹列表
    folders = [f for f in os.listdir(evaluation_dir) if os.path.isdir(os.path.join(evaluation_dir, f)) and f != "merge"]
    # 收集所有文件夹中文件的路径
    file_paths = defaultdict(list)  # {文件名: [文件路径1, 文件路径2, ...]}
    # 合并不同文件夹下的同名文件
    for folder in folders:
        for file_name in os.listdir(os.path.join(evaluation_dir, folder)):
            file_path = os.path.join(evaluation_dir, folder, file_name)
            if os.path.isfile(file_path):
                file_paths[file_name].append(file_path)
    
    for file_name, paths in file_paths.items():
        if len(paths) > 1:
            merged_item = dict()
            for file_path in paths:
                data = load_json_data(file_path)
                for item in data:
                    head_id, relation, tail_id = item["head_id"], item["relation"], item["tail_id"]
                    key = f"{head_id}|{relation}|{tail_id}"
                    if key not in merged_item:
                        # merged_item[key] = dict()
                        merged_item[key] = item.copy()
                        try:
                            del merged_item[key]["answer"]
                            del merged_item[key]["response"]
                            del merged_item[key]["prompt"]
                            del merged_item[key]["sentence"]
                        except Exception:
                            pass
                    model = file_path.split("/")[4]
                    try:
                        merged_item[key][f"{model}_answer"] = item["answer"]
                        if wiki:
                            flag = model.split('-')[-1] == "wiki"
                        if flag:
                            merged_item[key]["sentence"] = item["sentence"]
                            merged_item[key]["prompt"] = item["prompt"] 
                        # merged_item[key]["human"] = ""
                    except Exception:
                        label_mapping = {
                            "Entailment": "Correct",
                            "Contradiction": "Wrong"
                        }
                        merged_item[key][f"{model}_answer"] = label_mapping.get(item["FS_label"], item["FS_label"])
                        merged_item[key]["FS_label"] = item["FS_label"]
                        # print(f"file: {file_path}, eval_model: {model}, item: {item}")
                        # assert False
                    
            merged_list = process_merge(merged_item)
            save_json_data(os.path.join(save_dir, file_name), merged_list)

    

def select_distinct_merge():
    '''
    从merge的结果中挑出不同模型输出不一致的情况
    '''
    merge_dir = os.path.join(args.evaluation_dir, args.eval_type, "merge")   # 与merge_result()中的save_dir定义相同
    for file in os.listdir(merge_dir):
        path = os.path.join(merge_dir, file)
        data = load_json_data(path)
        distinct_list = list()
        for item in data:
            answer_values = [value.lower().strip() for key, value in item.items() if key.endswith("_answer")]
            if len(set(answer_values)) != 1:    #不同模型预测的结果不同
                distinct_list.append(item)
        if distinct_list:
            save_json_data(os.path.join(merge_dir, "distinct", file), distinct_list)


if __name__ == "__main__":
    # judge()
    # calculate()
    # process_other()
    merge_result(wiki=True)
    # select_distinct_merge()


            

