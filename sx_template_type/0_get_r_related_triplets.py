import argparse
import json
import openai
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='WN18RR')
args = parser.parse_args()


def get_r_triplets(tri_list,relation):
    filtered_triples = [(h, r, t) for (h, r, t) in tri_list if r == relation]
    return filtered_triples

def read_file_to_dict(file_path):
    # 创建一个空字典来存储结果
    dic = {}

    # 打开文件，读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 移除行尾的换行符，并分割MID和实体名称
            mid, entity = line.strip().split('\t')
            # 将MID和实体名称添加到字典中
            dic[mid] = entity
    return dic

def read_wn_entity_json_to_dict(wn_file_path):
    # 创建一个空字典来存储结果
    dic = {}
    with open(wn_file_path, 'r', encoding='utf-8') as file:
        entity_dic = json.load(file)
    for item in entity_dic:
        dic[item['entity_id']]=item['entity']
    return dic

#调用GPT 获得模板

#重新定义url 和key
def prepare_for_inference():
    client = openai.OpenAI(
        # api_key="sk-VlOkD9g9mwYHvtYNA047Aa77CbEb4623A49bD53c24086781",  # this is also the default, it can be omitted
        # base_url="https://guoke.huiyan-ai.com/v1/"
        api_key="sk-Ycy7gxsn40RIrn3TwZAypTgcPD0LnH01BlwMUMzJzhiNkz6y",
        base_url='https://api.chatanywhere.com.cn/v1'
    )
    return client

##调用gpt模型接口
def request_result_single(conversation,n_generate_sample):
    my_client=prepare_for_inference()
    response = my_client.chat.completions.create(
        model="gpt-4o-mini",  #选择的GPT模型名称
        messages=conversation,
        temperature=0,
        n = n_generate_sample  #生成的结果
    )
    results=[]
    for choice in response.choices:
        results.append(choice.message.content)
    return results

def reasoning_standard(prompt_text,n_generate_sample):
    system_role = dict(
        {'role': 'system',
         "content": "You are ChatGPT, a large language model trained by OpenAI. "
                    "\n Knowledge cutoff: 2021-09 \n Current date: 2023-08"})
    # prompt_text =  prompt_cs_type_reasoning(rela,rela_array)
    conversation = [system_role]
    conversation.append(
        {'role': 'user',
         "content": prompt_text})
    results = request_result_single(conversation, n_generate_sample)
    answer = results
    return answer

def construct_generation_prompt(demonstrations_for_r, r_name):
    gen_instruction = f'''
    "You are an excellent assistant when it comes to reading, comprehending, and summarizing.\n
    {demonstrations_for_r}\n
    The above triplet is a factual triplet from freebase. In above examples, What do you think \"{r_name}\" mean? \n
    Summarize and descript its meaning using the format: 
    \"If the example shows something (A , {r_name} , B) , it means A [mask] B.\" 
    Fill the mask and the statement should be as short as possible.'''

    return gen_instruction


def construct_generation_prompt_WN(demonstrations_for_r, r_name):
    gen_instruction = f'''
    "You are an excellent assistant when it comes to reading, comprehending, and summarizing.\n
    {demonstrations_for_r}\n
    The above triplet is a factual triplet from WN18RR datasets. In above examples, What do you think \"{r_name}\" mean? \n
    Summarize and descript its meaning using the format: 
    \"If the example shows something (A , {r_name} , B) , it means A [mask] B.\" 
    Fill the mask and the statement should be as short as possible.'''

    return gen_instruction

if __name__=='__main__':
    
    train_triplet = []
    #获取实体id和实体名称的映射
    # ent_dic=read_file_to_dict('/home/kayla/lhy/code/multi-source/data/' + args.dataset + '/FB15k_mid2name.txt')
    ent_dic=read_wn_entity_json_to_dict('/home/kayla/lhy/code/multi-source/data/' + args.dataset + '/entities.json')
    
    #读取train文件，获得关系为r的三元组。
    for line in open('/home/kayla/lhy/code/multi-source/data/' + args.dataset + '/train.txt', 'r'):
        head, relation, tail = line.strip('\n').split()
        train_triplet.append((ent_dic[head], relation, ent_dic[tail]))

    #遍历所有的关系：
    with open('/home/kayla/lhy/code/multi-source/data/' + args.dataset + '/relations.json', 'r', encoding='utf-8') as file:
        # # 读取文件内容为字符串
        # file_content = file.read()
        # 使用json.loads函数将字符串解析为Python数据结构
        relation_dic = json.load(file)
    # 获取所有的键并将它们转换为列表
    rela_list = list(relation_dic.keys())
    
    re_temp={}
    for re in tqdm(rela_list):
        # given_r_name="/location/country/form_of_government"   #给定一个关系名称
        given_r_name=re
        rule_demonstrations_for_r=get_r_triplets(train_triplet, given_r_name)

        gen_prompt_r=construct_generation_prompt_WN(rule_demonstrations_for_r[:10],given_r_name) #先粗糙进性生成
        res_new_rule_1=reasoning_standard(gen_prompt_r,1)
        re_temp[re]= res_new_rule_1[0]
    
    # 使用with语句打开文件，确保文件会被正确关闭
    with open('/home/kayla/lhy/code/multi-source/data/' + args.dataset + '/relations2_newTemplate.json', 'w', encoding='utf-8') as file:
        # 使用json.dump()函数将字典写入文件
        json.dump(re_temp, file, ensure_ascii=False, indent=4)
    
    print('---finish---')
    
    
    # A is a hypernym of B
    # it means A is a derived form of B."
    # A is an instance of B
    # A is related to or similar to B
    #  A is a member of B
    # A belongs to the domain of B.
    # A includes B as a part
    # A is used within the domain of B
    # A is associated with the region of B.
    # A is a variant or related form of B.
    # A is similar to B