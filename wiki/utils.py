import re
import pandas as pd
import json
from collections import Counter
from FlagEmbedding import FlagLLMReranker

reranker = FlagLLMReranker('/home/why/hfmodels/BAAI/bge-reranker-v2-m3', use_fp16=True,device='cuda:0')
def get_top_bgescore(target_text,text_list):
    target = []
    target.append(target_text)
    sentence_pairs = [[i,j] for i in target for j in text_list]
    scores = reranker.compute_score(sentence_pairs) # weights_for_different_modes(w) is used to do weighted sum: w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score
    return scores

def get_top_similar_sentences(sentences,bgescore,n):
    pd_data = pd.DataFrame({'sentence':sentences,'bgescore':bgescore})
    pd_data_sort = pd_data.sort_values(by='bgescore', ascending=False)
    if n<pd_data.shape[0]:
        pd_data_select = pd_data_sort.head(n)
    else:
        pd_data_select = pd_data_sort
    select_sentence = pd_data_select['sentence'].to_list()
    return select_sentence


def extract_counterspeech(text):
    marker = "[Counterspeech]:"
    start_index = text.find(marker)
    if start_index != -1:
        return text[start_index + len(marker):].strip()
    return ""

def check_hatespeech_in_json_list(json_list, target_sentence):
    for json_obj in json_list:
        if "hatespeech" in json_obj and target_sentence in json_obj["hatespeech"]:
            return True
    return False


def check_hatespeech_in_json_list_counter(json_list, target_sentence):
    for json_obj in json_list:
        if "hatepseech" in json_obj and target_sentence in json_obj["hatepseech"]:
            return True
    return False

def read_jsonlines(file_path):
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            json_list.append(json_obj)
    return json_list


def append_to_jsonlines(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def most_frequent_element(lst):
    # Create a Counter object from the list
    count = Counter(lst)
    # Find the element with the highest frequency
    most_common_element = count.most_common(1)
    return most_common_element[0][0] if most_common_element else None


def extract_vote_number(text):
    # Use regular expression to find the number within square brackets
    match = re.search(r'\[(\d+)\]', text)
    if match:
        return int(match.group(1))
    return None

def extract_claims(text):
    # 使用正则表达式提取以数字开头的行
    claims = re.findall(r'\d+\.\s+(.*)', text)
    return claims

def extract_queries(text):
    # 使用正则表达式提取以数字开头的行
    queries = re.findall(r'\d+\.\s+"([^"]+)"', text)
    return queries


def cut_sent(sentence):
    phrases = re.split(r'[.!?;]\s*', sentence)
    # Remove any empty strings from the list
    phrases = [phrase for phrase in phrases if phrase]
    return phrases

def filter_evidence(claim,evidence_list,top_n,reranker):
    target = []
    target.append(claim)
    # evidence_sentence = []
    # for evidence in evidence_list:
    #     evidence_sentence.extend(cut_sent(evidence))
    sentence_pairs = [[i,j] for i in target for j in evidence_list]
    scores = reranker.compute_score(sentence_pairs, normalize=True)
    evidence_score_pd = pd.DataFrame({"evidence":evidence_list,"score":scores})
    evidence_score_pd.sort_values(by="score", ascending=False, inplace=True)
    evidence_score_select = evidence_score_pd.head(top_n)
    clean_evidence = evidence_score_select["evidence"].tolist()
    return clean_evidence

def save_json_to_file(data, filename):
    """
    将JSON数据保存到文件

    参数：
    data (list): JSON数据组成的列表
    filename (str): 保存的文件名

    返回：
    None
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_counterspeech(text):
    # 使用正则表达式匹配 [Counterspeech] 和后续的内容
    match = re.search(r'\[Counterspeech\]: "(.*?)"', text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None








# def construct_json_data(hatespeech, claim_list, query_list, wiki_list):
#     json_data = {}
#     json_data['hatespeech'] = hatespeech
#     for i in range(len(claim_list)):
#         json_claim = {"claim_id": i,"claim_text":claim_list[i]}
#     json_data['Claims'] = json_claim
#     for i in range(len(query_list)):
#         json_query = {"query_id": i, "query_text": query_list[i]}
#     for i in range(len(wiki_list)):
#         json_wiki = wiki_list[]

# text = """Here are some counter-claims to refute the statement:
#
# [Claims]:
# 1. Asylum seekers and refugees do not automatically qualify for free council housing.
# 2. The allocation of council housing is subject to strict eligibility criteria and availability, not solely based on asylum seeker or refugee status.
# 3. In the UK, asylum seekers and refugees are typically housed in accommodation provided by the Home Office, not by local councils.
# 4. Council housing is generally prioritized for vulnerable groups, such as the homeless, families, and those with disabilities, not solely for asylum seekers and refugees.
# 5. The UK's asylum system and housing policies are designed to support those in need, but the allocation of resources is limited and subject to various factors, including funding and availability."""
#
# claims_list = extract_claims(text)
# print(claims_list)