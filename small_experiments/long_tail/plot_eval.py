import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys 
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, check_dir
from config import args

def sort(data, field=None):
    '''
    data: dict()，按照key的大小升序排列 {"70-80": {"mean_hits_10": 0.3362938740121135}}
    '''
    # 按照键的第一个数字进行排序
    sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('-')[0]))

    # 提取排序后的 mean_hits_10 值
    try:
        sorted_list = [data[key][field] for key in sorted_keys]
    except:
        sorted_list = [data[key] for key in sorted_keys]
    return sorted_list

def construct_data(bins, entity_count_dict, eval_data_llm, eval_data_transE, eval_data_simkgc, degree=False):
    '''
    bins: list(), eg. max_value = max(test_entity_count.values())
    bins = [0, 10, 20, 30, 40, 50, 100, max_value]

    entity_count_dict: dict()，实体出现频率
    eval_data_llm: dict() llm评估结果
    eval_data_transE: dict() transE评估结果
    eval_data_simkgc: dict() simkgc评估结果
    degree表示是否使用单个度数作为横轴
    '''
    all_dict = dict()
    
    if degree:
        all_dict["degree_intervals"] = [f"{bins[i]}" for i in range(len(bins))]
        bin_counts = {f"{bins[i]}": 0 for i in range(len(bins))}
        for value in entity_count_dict.values():
            for i in range(len(bins)):
                if bins[i] == value:
                    bin_counts[f"{bins[i]}"] += 1
                    break
    else:
        # 定义区间范围，按照eval.json中的范围定义的
        all_dict["degree_intervals"] = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]
        # 自动替换最后一个标签的上限为 ∞
        # 检查最后一个标签是否是一个区间，并将右边界替换为 ∞
        last_interval = all_dict["degree_intervals"][-1]
        match = re.match(r'(\[\d+,\d+\))', last_interval)
        if match:
            # 提取区间的开始部分，并替换右边界为 ∞
            all_dict["degree_intervals"][-1] = re.sub(r',\d+\)', ',+∞)', last_interval)
        bin_counts = {f"{bins[i]}-{bins[i+1]-1}": 0 for i in range(len(bins) - 1)}
            
        # 统计每个区间内的key数量
        for value in entity_count_dict.values():
            for i in range(len(bins) - 1):
                if bins[i] <= value < bins[i+1] and not degree:        #[0,10)
                    bin_counts[f"{bins[i]}-{bins[i+1]-1}"] += 1
                    break
    
    all_dict["bin_counts"] = sort(bin_counts)

    all_dict["mean_mrr_llm"] = sort(eval_data_llm, field="mean_mrr")
    all_dict["mean_hits_1_llm"] = sort(eval_data_llm, field="mean_hits_1")
    all_dict["mean_hits_3_llm"] = sort(eval_data_llm, field="mean_hits_3")
    all_dict["mean_hits_10_llm"] = sort(eval_data_llm, field="mean_hits_10")

    all_dict["mean_mrr_transE"] = sort(eval_data_transE, field="mean_mrr")
    all_dict["mean_hits_1_transE"] = sort(eval_data_transE, field="mean_hits_1")
    all_dict["mean_hits_3_transE"] = sort(eval_data_transE, field="mean_hits_3")
    all_dict["mean_hits_10_transE"] = sort(eval_data_transE, field="mean_hits_10")

    all_dict["mean_mrr_simkgc"] = sort(eval_data_simkgc, field="mean_mrr")
    all_dict["mean_hits_1_simkgc"] = sort(eval_data_simkgc, field="mean_hits_1")
    all_dict["mean_hits_3_simkgc"] = sort(eval_data_simkgc, field="mean_hits_3")
    all_dict["mean_hits_10_simkgc"] = sort(eval_data_simkgc, field="mean_hits_10")
    
    return all_dict



def plot(all_dict, path, k=1):
    '''
    k=1,3,10
    '''

    # 创建图形和第一个y轴
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制柱状图
    x = np.arange(len(all_dict["degree_intervals"]))
    bar_width = 0.6
    ax1.bar(x, all_dict["bin_counts"], color='#f3d0a4', width=bar_width, label='Frequency')


    ax1.set_xlabel('Entity Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(x)
    # ax1.set_xticklabels(all_dict["degree_intervals"])
    ax1.set_xticklabels(all_dict["degree_intervals"], rotation=45, ha="right")  # 旋转标签

    # 创建第二个y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Hits@{k}')

    # 绘制不同模型的Hits@1折线图
    ax2.plot(x, all_dict[f"mean_hits_{k}_llm"], color='#c7522a', marker='o', label='MuSKGC', linewidth=3.0, markersize=9)
    ax2.plot(x, all_dict[f"mean_hits_{k}_transE"], color='#74a892', marker='s', label='TransE', linewidth=3.0, markersize=9)
    ax2.plot(x, all_dict[f"mean_hits_{k}_simkgc"], color='#6886af', marker='*', label='SimKGC', linewidth=3.0, markersize=12)
    ax2.set_ylim(0.0, 0.8)  # 设置Hits@1的y轴范围

    # 添加图例
    fig.legend(loc='upper left', bbox_to_anchor=(0.10, 0.95))
    plt.tight_layout()  # 自动调整布局
    plt.show()
    plt.savefig(path)


if __name__ == "__main__":
    entity_frequency = load_json_data(args.entity_frequency_path)

    if args.degree:
        eval_data_llm = load_json_data(f"small_experiments/long_tail/{args.task}/llm/eval_degree.json")
        eval_data_transE = load_json_data(f"small_experiments/long_tail/{args.task}/transE/eval_degree.json")
        eval_data_simkgc = load_json_data(f"small_experiments/long_tail/{args.task}/simkgc/eval_degree.json")
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        all_dict = construct_data(bins, entity_frequency, eval_data_llm, eval_data_transE, eval_data_simkgc, degree=args.degree)
        plot(all_dict, os.path.join(args.long_tail_image_degree_dir, "image_hits_1.png"), k=1)
    else:
        eval_data_llm = load_json_data(f"small_experiments/long_tail/{args.task}/llm/eval.json")
        eval_data_transE = load_json_data(f"small_experiments/long_tail/{args.task}/transE/eval.json")
        eval_data_simkgc = load_json_data(f"small_experiments/long_tail/{args.task}/simkgc/eval.json")
        max_value = max(entity_frequency.values())
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, max_value+1]
        all_dict = construct_data(bins, entity_frequency, eval_data_llm, eval_data_transE, eval_data_simkgc)
        plot(all_dict, os.path.join(args.long_tail_image_dir, "image_hits_1.png"), k=1)
