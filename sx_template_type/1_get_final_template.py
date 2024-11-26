import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='WN18RR')
args = parser.parse_args()

with open('/home/kayla/lhy/code/multi-source/data/' + args.dataset + '/relations2_newTemplate.json', 'r', encoding='utf-8') as file:
        # # 读取文件内容为字符串
        # file_content = file.read()
        # 使用json.loads函数将字符串解析为Python数据结构
        relation_dic = json.load(file)

final_template={}
for key, value in relation_dic.items():
    # 找到 "it means" 的位置
    start_index = value.find("it means") + len("it means")
    # 提取 "it means" 后面的内容
    meaning = value[start_index:]
    # 将 A 替换为 H
    meaning = meaning.replace("A", "H")
    # 将 B 替换为 T
    meaning = meaning.replace("B", "T")
    final_template[key]=meaning

with open('/home/kayla/lhy/code/multi-source/data/' + args.dataset + '/relations2_finalTemplate.json', 'w', encoding='utf-8') as file:
        # 使用json.dump()函数将字典写入文件
        json.dump(final_template, file, ensure_ascii=False, indent=4)   
print('----final finish---')