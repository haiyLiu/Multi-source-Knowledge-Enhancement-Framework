# 1. 统计test数据集中实体在train中出现的频率（统计入度+出度），根据总度数挑出长尾实体；画个实体度数分布图


import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, load_txt_data
from config import args

def static_frequency(data):
    '''
    根据数据data统计实体出现的频率
    data: list()
    '''

    # 初始化字典用于记录每个实体的出现次数
    entity_count = {}

    # 统计实体出现频率
    for line in data:
        # 将每行分割成实体和关系
        h_id, r, t_id = line.split("\n")[0].split("\t")
        
        # 统计每个实体的频率，按照id统计
        for id in [h_id, t_id]:
            if id not in entity_count:
                entity_count[id] = 0
            entity_count[id] += 1

    return entity_count # dict()

def static_test_from_train(test_data, train_entity_count):
    '''
    统计test数据集中实体在train中出现的频率
    '''
    # 初始化字典用于记录每个实体的出现次数
    entity_count = {}

    # 统计实体出现频率
    for line in test_data:
        # 将每行分割成实体和关系
        h_id, r, t_id = line.split("\n")[0].split("\t")
        for id in [h_id, t_id]:
            if id not in train_entity_count:    #test中有的实体，但train中没有
                entity_count[id] = 0
            else:
                entity_count[id] = train_entity_count[id]
    return entity_count

def id2name(data, entity_id2name):
    '''
    data: dict(), key=id, value=count of entity
    '''
    new_data = dict()
    for key,value in data.items():
        entity_name = entity_id2name[id]
        if entity_name not in new_data:
            new_data[entity_name] = 0
        new_data[entity_name] += value
    return new_data

def plot_fig(entity_count_dict, path):
    '''
    不设置区间
    '''
    # 统计每个 value 值对应的 key 的数量
    value_counts = Counter(entity_count_dict.values())

    # 将数据拆分为 x 和 y 轴
    x_values = sorted(value_counts.keys())
    y_counts = [value_counts[x] for x in x_values]

    # 绘制曲线图
    plt.figure(figsize=(12, 8))
    # plt.plot(x_values, y_counts, label='Entity Frequency', linewidth=3, marker='o')
    # plt.xlabel("Degree")
    # plt.ylabel("Count of Entities")

    plt.plot(y_counts, x_values, label='Entity Frequency', linewidth=3, marker='o')
    plt.xlabel("Count of Entities")
    plt.ylabel("Degree")

    plt.title(f"{args.task}")
    plt.legend()
    plt.show()

    # 保存图片
    plt.savefig(path)

def plot_fig_bins(entity_count_dict, path, bins):
    '''
    entity_count_dict: dict() ={entity: 出现频率}
    n: int 表示分出的区间个数
    绘制一个横轴为出现频率，纵轴为出现频率对应的实体数量的分布图
    '''

    # 定义区间范围
    bin_counts = {f"{bins[i]}-{bins[i+1]-1}": 0 for i in range(len(bins) - 1)}

    # 统计每个区间内的key数量
    for value in entity_count_dict.values():
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i+1]:
                bin_counts[f"{bins[i]}-{bins[i+1]-1}"] += 1
                break

    # 准备绘图数据
    x_labels = list(bin_counts.keys())
    y_counts = list(bin_counts.values())

    # 绘制曲线图
    plt.figure(figsize=(12, 8))
    plt.plot(x_labels, y_counts, marker='o', label='Entity Frequency', linewidth=2.5)
    plt.xlabel("Degree")
    plt.ylabel("Count of Entities")
    plt.title(f"{args.task}")
    plt.legend()
    plt.show()

    # 保存图片
    plt.savefig(path)

def plot_fig_bar(entity_count_dict, path, bins, interval_labels):
    '''
    绘制区间直方图
    '''
    # 统计每个 value 值对应的 key 的数量
    value_counts = Counter(entity_count_dict.values())

    # 将 value 分为 5 个区间
    interval_counts = [0] * (len(bins) - 1)

    # 对统计数据进行分区
    for value, count in value_counts.items():
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                interval_counts[i] += count
                break

    # 绘制直方图
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(interval_counts)), interval_counts, tick_label=interval_labels, color='#A3B7CA', label='Entity Frequency', width=0.5)
    plt.xlabel("Degree")
    plt.ylabel("Entity Frequency")
    plt.title(f"{args.task}")
    # plt.legend()
    plt.show()

    # 保存图片
    plt.savefig(path)

def plot_long_tail(entity_count, degree):
    '''
    求出value小于等于degree的项
    '''
    # 使用字典推导式过滤小于阈值的项
    filtered_data = {k: v for k, v in entity_count.items() if v <= degree}

    # 绘制曲线图
    plot_fig(filtered_data, os.path.join(args.image_dir,"image_filter.jpg"))

    # 绘制直方图
    max_value = max(entity_count.values())
    bins = [0, 10, 20, 30, 40, 50, 100]
    interval_labels = ['[0,10]', '[10,20]', '[20,30]', '[30,40]', '[40,50]', '[50,100]']
    plot_fig_bar(filtered_data, os.path.join(args.image_dir, "image_filter_bar.jpg"), bins, interval_labels)
    
    return filtered_data

if __name__ == "__main__":
    entity_id2name = load_json_data(args.entity_id2name_path)
    test_data = load_txt_data(args.test_txt_path)
    train_data = load_txt_data(args.train_txt_path)

    train_entity_count = static_frequency(train_data)
    test_entity_count = static_test_from_train(test_data, train_entity_count)

    # 按照值降序排序
    sorted_dict = dict(sorted(test_entity_count.items(), key=lambda item: item[1], reverse=False))
    image_dir = args.image_dir
    # plot_fig(test_entity_count, os.path.join(image_dir, "image_1_reverse.jpg"))

    # max_value = max(test_entity_count.values())
    # bins = [0, 10, 20, 30, 40, 50, 100, max_value]
    # interval_labels = ['[0,10]', '[10,20]', '[20,30]', '[30,40]', '[40,50]', '[50,100]', '[100,max]']
    # plot_fig_bins(test_entity_count, os.path.join(image_dir, "image_2.jpg"), bins)
    

    # plot_fig_bar(test_entity_count, os.path.join(image_dir, "image_3.jpg"), bins, interval_labels)

    # 绘制度数<=degree的图
    # plot_long_tail(test_entity_count, degree=100)

    #存储test的实体频率数据
    save_json_data(args.entity_frequency_path, sorted_dict)

