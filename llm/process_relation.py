import os
import re
import json
from collections import Counter

from llm import MyLLM
from final_llm import load_data, process_answer
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, check_dir
from config import args

def load_infomation(neighbors_info, wikibase_info, entity, relation, query, candidate_entities, inverse=False):

    task_desc = "You are a good assistant to perform entity prediction. Given a goal question and a list of candidate answers to this question. You need to order these candidate answers in the list to let candidate answers which are more possible to be the answer to the question prior. Meanwhile, I hope that based on the knowledge I've provided, you can help me predict some possible new entities that don't appear among the candidate answers."

    

    kg_info = f"**Knowledge from the knowledge graph**: {neighbors_info}."
    wiki_text = [wikibase_info[i]["text"] for i in range(len(wikibase_info))]
    wiki_info = f"**Knowledge from Wiki knowledge base**: {wiki_text}."

    if not inverse: # forward predict
        triplet = f"({entity}, {relation}, [MASK])"
    else:
        triplet = f"([MASK], {relation}, {entity})"

    query_prompt = f"The goal question is: predict the tail entity [MASK] from the given {triplet} by completing the sentence '{query}'. The list of candidate answers is {candidate_entities}. Please note that in the list of candidate answers, each '[ENTITY]' represents an entity, please understand it as a whole. I hope you can help me complete two tasks:\n(1) please sort the candidate answers based on the knowledge mentioned above and your knowledge. The output format of the sorted candidate answers is strictly in accordance with <The list of sorted candidate answers is [answer_1 | answer_2 | ... | answer_20]>. Note your output sorted order should contain all the candidates in the list but not add new answers to it, so please check the list of sorted candidate answers to ensure that it is 20 in length.\n (2)please predict some possible new entities that don't appear in the candidate answers based on the knowledge mentioned above and your knowledge.  The output format of the predicted new entities is strictly in accordance with <The predicted new entities are [entity_1 | entity_2 | ... | entity_10]>."

    conversation = [
        {"role": "system", "content": task_desc},
        {"role": "assistant", "content": kg_info},
        {"role": "assistant", "content": wiki_info},
        {"role": "user", "content": query_prompt},
    ]

    prompt = f"{task_desc}\n {kg_info}\n {wiki_info}\n {query_prompt}"
    return prompt, conversation

def query_llm(conversation, llm, model="gpt-4o-mini"):
    '''
    prompt : list()
    '''
    response = llm.client.chat.completions.create(
        model=model,  #选择的GPT模型名称
        messages=conversation,
    )
    answer = response.choices[0].message.content
    return answer

def update_query(query_template_data, item):
    '''
    根据query模板更新query，用于后续LLM的查询
    query_template_data : dict()
    item : dict()
    '''
    query_template_t = query_template_data[item["relation"]]["query_t"]
    query_template_h = query_template_data[item["relation"]]["query_h"]
    query_t = query_template_t.replace('[H]', item["head"])
    query_h = query_template_h.replace('[T]', item["tail"])
    return query_t, query_h

def check_relation(relation, data):
    '''
    检查某个文件是否存在某种关系
    relation: str
    data: list(dict())
    '''
    for item in data:
        if item["relation"].strip().lower() == relation.strip().lower():
            return True
    return False

def statistic_relation_frequency(relation, data):
    '''
    统计关系在某个文件中出现的频率
    relation: str
    data: list(dict())
    '''
    relation_list = [item["relation"] for item in data]
    relation_counts = Counter(relation_list)
    times = relation_counts[relation]
    return times

if __name__ == "__main__":
    llm = MyLLM()


    # 预测某种关系出现在哪些文件中
    r = "nutrient nutrition fact nutrients food "
    LLM_result_dir = os.path.join(args.LLM_result_dir, args.llm_model)
    sum_times = 0
    for file in sorted(os.listdir(LLM_result_dir)):
        path = os.path.join(LLM_result_dir, file)
        data = load_json_data(path)
        sum_times += statistic_relation_frequency(r, data)
        if check_relation(r, data):
            print(file)
    print(f"times: {sum_times}")
        

    '''
    # 针对根据关系新挑出来的模板，只处理这一部分的三元组
    query_template_path = "data/FB15k237/query_template_2.json"
    query_template_data = load_json_data(query_template_path)

    begin = 151
    end = 205
    test_files = ["test_" + str(i) + "_gpt4omini_allinfo.json" for i in range(begin, end+1)]
    answer_files = ["test" + str(i) + "_answer.json" for i in range(begin, end+1)]
    answer_dir = os.path.join(args.LLM_result_dir, args.llm_model)


    for test_file, answer_file in zip(test_files, answer_files):
        test_path = os.path.join(args.test_dir, test_file)
        test_data = load_json_data(test_path)

        answer_path = os.path.join(answer_dir, answer_file)
        answer_data = load_json_data(answer_path)

        new_list = list()
        for i in range(len(answer_data)):
            test_item = test_data[i]
            answer_item = answer_data[i]
            if answer_item["relation"] in query_template_data:
                head, relation, tail, query_t, query_h, forward_candidate_entities, backward_candidate_entities, h_neighbors_info, t_neighbors_info, forward_wiki_info, backward_wiki_info = load_data(test_item)

                query_t, query_h = update_query(query_template_data, test_item)

                forward_prompt, forward_conversation = load_infomation(h_neighbors_info, forward_wiki_info, head, relation, query_t, forward_candidate_entities)
                forward_answer = query_llm(forward_conversation, llm, model=args.llm_model)
                forward_sorted_candidates, forward_predicted_entities = process_answer(forward_answer)
                answer_item["forward_prompt"] = forward_prompt
                answer_item["forward_sorted_candidates"] = forward_sorted_candidates
                answer_item["forward_predicted_entities"] = forward_predicted_entities

                backward_prompt, backward_conversation = load_infomation(t_neighbors_info, backward_wiki_info, tail, relation, query_h, backward_candidate_entities)
                backward_answer = query_llm(backward_conversation, llm, model=args.llm_model)
                backward_sorted_candidates, backward_predicted_entities = process_answer(backward_answer)
                answer_item["backward_prompt"] = backward_prompt
                answer_item["backward_sorted_candidates"] = backward_sorted_candidates
                answer_item["backward_predicted_entities"] = backward_predicted_entities

            new_list.append(answer_item)
        save_json_data(answer_path, new_list)
    '''
                