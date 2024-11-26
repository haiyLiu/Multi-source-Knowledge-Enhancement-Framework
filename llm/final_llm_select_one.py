import os
import re
import json


from llm import MyLLM
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
# from llm.llm import MyLLM
from base_methods import load_json_data, save_json_data, check_dir
from config import args


def load_data(item):
    '''
    item : dict()
    '''
    head, relation, tail = item["head"], item["relation"], item["tail"]
    query_t, query_h = item["query_t"], item["query_h"]
    forward_candidate_entities = item["forward_topk_ents"]
    backward_candidate_entities = item["backward_topk_ents"]
    if args.use_filter_candidates:
        forward_candidate_entities = item["forward_filter_candidates"]
        backward_candidate_entities = item["backward_filter_candidates"]
    h_neighbors = item["h_neighbors"] if "h_neighbors" in item else ""
    h_neighbors_info = item["h_neighbors_info"]
    t_neighbors = item["t_neighbors"] if "t_neighbors" in item else ""
    t_neighbors_info = item["t_neighbors_info"]
    forward_wiki_info = item["forward_wiki_info"]
    backward_wiki_info = item["backward_wiki_info"]
    forward_source_rank = item["forward_ground_truth_rank"]
    backward_source_rank = item["backward_ground_truth_rank"]
    return head, relation, tail, query_t, query_h, forward_candidate_entities, backward_candidate_entities,h_neighbors, h_neighbors_info, t_neighbors, t_neighbors_info, forward_wiki_info, backward_wiki_info, forward_source_rank, backward_source_rank

def save_data(item, forward_prompt, backward_prompt, forward_sorted_candidates, backward_sorted_candidates, forward_source_rank, backward_source_rank, forward_llm_rank, backward_llm_rank):
    answer = dict()
    answer["head_id"] = item["head_id"]
    answer["head"] = item["head"]
    answer["relation"] = item["relation"]
    answer["tail_id"] = item["tail_id"]
    answer["tail"] = item["tail"]

    answer["forward_source_rank"] = forward_source_rank
    answer["forward_llm_rank"] = forward_llm_rank
    answer["backward_source_rank"] = backward_source_rank
    answer["backward_llm_rank"] = backward_llm_rank

    answer["forward_prompt"] = forward_prompt
    answer["forward_sorted_candidates"] = forward_sorted_candidates
    

    answer["backward_prompt"] = backward_prompt
    answer["backward_sorted_candidates"] = backward_sorted_candidates
    
    return answer

def construct_select_prompt(neighbors, neighbors_info, wikibase_info, entity, relation, query, candidate_entities, inverse=False):
    task_desc = "# You are a good assistant to perform entity prediction. Given a goal question and a list of candidate answers to this question. You must select the most possibly correct answer to the question from the list of candidate answers."

    kg_info = f"**Knowledge from the knowledge graph**: {neighbors_info}."
    wiki_text = [wikibase_info[i]["text"] for i in range(len(wikibase_info))]
    wiki_info = f"**Knowledge from Wiki knowledge base**: {wiki_text}."


    if not inverse: # forward predict
        triplet = f"({entity}, {relation}, [MASK])"
    else:
        triplet = f"([MASK], {relation}, {entity})"

    query_prompt = f"# The goal question is: predict the tail entity [MASK] from the given {triplet} by completing the sentence '{query}'."

    candidate_prompt = f"# The list of candidate answers is {candidate_entities}."

    task_query = "I hope you can help me complete a task:\n(1) please select a correct candidate answer to above question from the list of candidate answers based on the knowledge mentioned above and your knowledge. The output format of the correct candidate answer is strictly in accordance with <The correct candidate answer is [entity]>."

    prompt = f"{task_desc}\n {kg_info}\n {wiki_info}\n {query_prompt}\n {candidate_prompt}\n {task_query}"

    if not args.use_relation_template:
        kg_info = f"**Knowledge from the knowledge graph**: {neighbors}."
        query_prompt = f"The goal question is: predict the tail entity [MASK] from the given {triplet}. The list of candidate answers is {candidate_entities}."
        prompt = f"{task_desc}\n {kg_info}\n {wiki_info}\n {query_prompt}\n {candidate_prompt}\n {task_query}"
    if not args.use_kg_neis:
        kg_info = ""
        prompt = f"{task_desc}\n {wiki_info}\n {query_prompt}\n {candidate_prompt}\n {task_query}"
    if not args.use_wiki_docs:
        wiki_info = ""
        prompt = f"{task_desc}\n {kg_info}\n {query_prompt}\n {candidate_prompt}\n {task_query}"

    conversation = [
        {"role": "system", "content": task_desc},
        {"role": "assistant", "content": kg_info},
        {"role": "assistant", "content": wiki_info},
        {"role": "assistant", "content": query_prompt},
        {"role": "assistant", "content": candidate_prompt},
        {"role": "user", "content": task_query},
    ]

    return prompt, conversation

def load_infomation(item):
    '''
    item : dict()
    '''
    head, relation, tail, query_t, query_h, forward_candidate_entities, backward_candidate_entities,h_neighbors, h_neighbors_info, t_neighbors, t_neighbors_info, forward_wiki_info, backward_wiki_info, forward_source_rank, backward_source_rank = load_data(item)

    new_h_neighbors = list()
    new_t_neighbors = list()
    if args.task == "WN18RR":
        head = ' '.join(head.split('_')[:-2])
        tail = ' '.join(tail.split('_')[:-2])
        for nei in h_neighbors:
            h, r, t = nei.split("\t")
            h = ' '.join(h.split('_')[:-2])
            t = ' '.join(t.split('_')[:-2])
            s = f"({h}, {r}, {t})"
            new_h_neighbors.append(s)
        for nei in t_neighbors:
            h, r, t = nei.split("\t")
            h = ' '.join(h.split('_')[:-2])
            t = ' '.join(t.split('_')[:-2])
            s = f"({h}, {r}, {t})"
            new_t_neighbors.append(s)
    else:
        for nei in h_neighbors:
            h, r, t = nei.split("\t")
            s = f"({h}, {r}, {t})"
            new_h_neighbors.append(s)
        for nei in t_neighbors:
            h, r, t = nei.split("\t")
            s = f"({h}, {r}, {t})"
            new_t_neighbors.append(s)
    
    forward_prompt, forward_conversation = construct_select_prompt(new_h_neighbors, h_neighbors_info, forward_wiki_info, head, relation, query_t, forward_candidate_entities)  #!use_relation_template=True表示不使用关系模板

    backward_prompt, backward_conversation = construct_select_prompt(new_t_neighbors, t_neighbors_info, backward_wiki_info, tail, relation, query_h, backward_candidate_entities, inverse=True)    #!use_relation_template=True表示不使用关系模板

    re_prompt = "Note your output sorted order should contain all the candidates in the list but not add new answers to it, so please check the list of sorted candidate answers to ensure that it is 20 in length."
    
    return forward_prompt, forward_conversation, backward_prompt, backward_conversation, re_prompt, head, tail, forward_source_rank, backward_source_rank, forward_candidate_entities, backward_candidate_entities


def query_llm(conversation, llm, model="gpt-4o-mini"):
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

def process_answer_select(answer, candidates):
    match = re.search(r'\[(.*?)\]', answer)
    if match:
        extracted_text = match.group(1)
        if extracted_text in candidates:
            candidates.remove(extracted_text)  # 先删除匹配的元素
            candidates.insert(0, extracted_text)  # 将元素插入到第一位
    return candidates


def llm_predict_rank(answer, entity):
    '''
    根据llm给出的排序列表，输出实体在列表中的排名，未匹配到输出-1
    answer: list()
    entity: str
    '''
    #!注意：因为给出的排序列表中有重复的实体名，所以应该取最高的rank
    rank = [index+1 for index, value in enumerate(answer) if value.lower().strip() == entity.lower().strip()]
    high_rank = min(rank) if rank else -1
    return high_rank


if __name__ == "__main__":
    # test_files = [test_file for test_file in os.listdir(args.test_dir) if test_file.endswith("_allinfo.json")]
    llm = MyLLM()

    begin = 1
    end = 1
    test_files = ["test_" + str(i) + "_gpt4omini_allinfo.json" for i in range(begin, end+1)]

    for test_file in test_files:
        llm_answer = list()
        test_path = os.path.join(args.test_dir, test_file)
        test_data = load_json_data(test_path)
        for item in test_data:
            forward_prompt, forward_conversation, backward_prompt, backward_conversation, re_prompt, head, tail, forward_source_rank, backward_source_rank, forward_candidate_entities, backward_candidate_entities = load_infomation(item) #!filter_candidates表示从候选实体中过滤train中三元组里出现过的

            forward_answer = query_llm(forward_conversation, llm, model=args.llm_model)
            forward_sorted_candidates = process_answer_select(forward_answer, forward_candidate_entities)
            if len(forward_sorted_candidates) == 0:
                forward_conversation.append({
                    'role': 'user',
                    "content": re_prompt
                })
                forward_answer = query_llm(forward_conversation, llm, model=args.llm_model)
                forward_sorted_candidates = process_answer_select(forward_answer, forward_candidate_entities)

            backward_answer = query_llm(backward_conversation, llm, model=args.llm_model)
            backward_sorted_candidates = process_answer_select(backward_answer, backward_candidate_entities)
            if len(backward_sorted_candidates) == 0:
                backward_conversation.append({
                    'role': 'user',
                    "content": re_prompt
                })
                backward_answer = query_llm(backward_conversation, llm, model=args.llm_model)
                backward_sorted_candidates = process_answer_select(backward_answer, backward_candidate_entities)

            forward_llm_rank = llm_predict_rank(forward_sorted_candidates, tail)
            backward_llm_rank = llm_predict_rank(backward_sorted_candidates, head)

            answer = save_data(item, forward_prompt, backward_prompt, forward_sorted_candidates, backward_sorted_candidates, forward_source_rank, backward_source_rank, forward_llm_rank, backward_llm_rank)
            llm_answer.append(answer)

        save_dir = os.path.join(args.LLM_result_dir,args.llm_model)
        check_dir(save_dir)
        seq_file = test_file.split("_")[0] + test_file.split("_")[1] + "_answer.json"
        save_json_data(os.path.join(save_dir, seq_file), llm_answer)


            