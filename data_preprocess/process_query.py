import json
import os
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from config import args

def load_json_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json_data(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def select_null_query(data, s):
    """
    data: list()
    """
    for item in data:
        relation = item["relation"]
        query_t = item["query_t"]
        query_h = item["query_h"]
        if query_t is None or query_h is None:
            s.add(relation)

def select_mask(data, s):
    """
    data: list()
    """
    for item in data:
        query_t = item["query_t"].lower()
        query_h = item["query_h"].lower()
        relation = item["relation"]
        if "[mask]" in query_t or "[mask]" in query_h:
            s.add(relation)

def process_query(data, query_template, new_list):
    """
    data: list()
    query_template: dict()
    """
    for item in data:
        relation = item["relation"]
        head = item["head"]
        tail = item["tail"]
        if args.task == "WN18RR":
            head = ' '.join(item["head"].split('_')[:-2])
            tail = ' '.join(item["tail"].split('_')[:-2])
        if relation in query_template:
            query_template_t = query_template[relation]["query_t"]
            query_template_h = query_template[relation]["query_h"]

            item["query_t"] = query_template_t.replace('[H]', head)
            item["query_h"] = query_template_h.replace('[T]', tail)
        new_list.append(item)

def find_same_query(data, s):
    for item in data:
        query_t = item['query_t']
        query_h = item['query_h']
        relation = item['relation']
        try:
            if query_t.lower() == query_h.lower():
                s.add(relation)
        except Exception:
            s.add(relation)

if __name__ == "__main__":
    file_dir = f"data/{args.task}/small_test"
    s = set()

    query_template_path = f"data/{args.task}/query_template.json"
    query_template = load_json_data(query_template_path)

    for file in os.listdir(file_dir):
        new_list = list()
        if file.endswith('gpt4omini_allinfo.json'):
            path = os.path.join(file_dir, file)
            data = load_json_data(path) #list()

            process_query(data, query_template, new_list)   # 根据关系模板增加query_t,query_h字段
            save_json_data(path, new_list)

            # find_same_query(data, s)
            # select_null_query(data, s)
            # select_mask(data, s)
            
    # print(s)
    # print(len(s))