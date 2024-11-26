import pandas as pd
from chat_with_llm import *
from prompt_set import *
from utils import *
from tqdm import trange
import os
from wiki.wiki_search import search_wiki_doc
from FlagEmbedding import FlagLLMReranker,FlagReranker


hatespeech_name = ['conan','mtconan','dynahate','ihchate','mtkgconan']
path = r"/home/why/ocean_projects/Fact_counterspeech/middle_resluts/ihchate_query.json"
save_path = r"/home/why/ocean_projects/Fact_counterspeech/middle_resluts/ihchate_evidence.json"

query_json_list= read_jsonlines(path)
wiki_json_all = []

hatespeech_list = []
for query_json in query_json_list:
    hatespeech_list.append(query_json['hatespeech'])

if os.path.exists(save_path):
    query_json_list_check = read_jsonlines(save_path)
else:
    query_json_list_check =[]

query_json_all =[]

for i in trange(len(query_json_list)):
    if check_hatespeech_in_json_list(query_json_list_check,query_json_list[i]["hatespeech"]):
        print("exist")
        continue
    wiki_doc_num = 5
    query_list = []

    if len(query_json_list[i]) == 7:
        for j in range(5):
            query_list.append(query_json_list[i]["query_"+str(j)])
        wiki_doc = search_wiki_doc(query_list, wiki_doc_num)
        wiki_json_single = {}
        wiki_json_single['hatespeech'] = query_json_list[i]["hatespeech"]
        wiki_json_single['claim'] = query_json_list[i]["claim"]

        for k in range(5):
            wiki_json_single['query_'+str(k)] = query_json_list[i]['query_'+str(k)]
            for l in range(5):
                wiki_json_single['title_' + str(k) + "_" + str(l)] = wiki_doc[k][l]["title"]
                wiki_json_single['evidence_' + str(k) + "_" + str(l)] = wiki_doc[k][l]["text"]
        wiki_json_all.append(wiki_json_single)
        print(wiki_json_single)
        if len(wiki_json_all)% 10 ==0:
            append_to_jsonlines(wiki_json_all,save_path)
            wiki_json_all = []
append_to_jsonlines(wiki_json_all,save_path)