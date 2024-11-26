import json
import os

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_txt_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data

def save_json_data(path, data):
    dir_path = path.rsplit("/", 1)[0]
    check_dir(dir_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def check_dir(dir_path):
    '''
    判断文件夹是否存在，不存在则创建
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def remove_duplicates(original_list):
    '''
    去除list中的重复元素，同时保持原有的顺序
    '''
    seen = set()  # 用于记录已经出现的元素
    result = []   # 用于存储去重后的结果

    for item in original_list:
        if item not in seen:
            seen.add(item)  # 将新元素添加到集合中
            result.append(item)  # 将新元素添加到结果列表中

    return result