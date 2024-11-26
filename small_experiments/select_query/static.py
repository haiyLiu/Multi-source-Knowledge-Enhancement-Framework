# 经过人工评判后的query，统计gpt和自动化方法的评估效果
import os
import re
import sys 
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from collections import defaultdict
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data
from config import args
from llm.llm import MyLLM


def static(data, gpt_wiki_counts, FS_counts, times):
    '''
    统计data中gpt和自动化方法的评估正确率
    data: list(dict())
    gpt_wiki_counts: int
    FS_counts: int
    '''
    for item in data:
        human_answer = item["human_answer"]
        gpt_wiki_answer = item["gpt-4o-wiki_answer"]
        FS_answer = item["FS_answer"]
        if gpt_wiki_answer.lower().strip() == human_answer.lower().strip():
            gpt_wiki_counts += 1
        if FS_answer.lower().strip() == human_answer.lower().strip():
            FS_counts += 1
    times += len(data)
    return gpt_wiki_counts, FS_counts, times

def calculate_f1_score(tp, tn, fp, fn):
    # 计算 Precision 和 Recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    # 计算 F1 Score
    if precision + recall == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def static_metrics(files):
    gpt_tp = gpt_tn = gpt_fp = gpt_fn = 0
    fs_tp = fs_tn = fs_fp = fs_fn = 0
    gpt_fs_tp, gpt_fs_tn = 0, 0
    gpt_fs_human_tp, gpt_fs_human_tn = 0, 0
    times = 0
    for file in files:
        path = os.path.join(merge_dir, file)
        data = load_json_data(path)
        for item in data:
            human_answer = item['human_answer']
            gpt_answer = item['gpt-4o-wiki_answer']
            fs_answer = item['FS_answer']
            
            # gpt-4o-wiki_answer 评估
            if human_answer == "Correct" and gpt_answer == "Correct":
                gpt_tp += 1
            elif human_answer == "Wrong" and gpt_answer == "Wrong":
                gpt_tn += 1
            elif human_answer == "Wrong" and gpt_answer == "Correct":
                gpt_fp += 1
            elif human_answer == "Correct" and gpt_answer == "Wrong":
                gpt_fn += 1
            
            # FS_answer 评估
            if human_answer == "Correct" and fs_answer == "Correct":
                fs_tp += 1
            elif human_answer == "Wrong" and fs_answer == "Wrong":
                fs_tn += 1
            elif human_answer == "Wrong" and fs_answer == "Neutral":
                fs_tn += 1
            elif human_answer == "Wrong" and fs_answer == "Correct":
                fs_fp += 1
            elif human_answer == "Correct" and fs_answer == "Wrong":
                fs_fn += 1
            elif human_answer == "Correct" and fs_answer == "Neutral":
                fs_fn += 1

            # 计算B交C
            if gpt_answer == "Correct" and fs_answer == "Correct":
                gpt_fs_tp += 1
            elif gpt_answer == "Wrong" and fs_answer == "Wrong":
                gpt_fs_tn += 1

            # 计算A交B交C
            if human_answer == "Correct" and gpt_answer == "Correct" and fs_answer == "Correct":
                gpt_fs_human_tp += 1
            elif human_answer == "Wrong" and gpt_answer == "Wrong" and fs_answer == "Wrong":
                gpt_fs_human_tn += 1


        times += len(data)
    # 计算 Precision 和 Recall
    gpt_precision = gpt_tp / (gpt_tp + gpt_fp) if (gpt_tp + gpt_fp) != 0 else 0
    gpt_recall = gpt_tp / (gpt_tp + gpt_fn) if (gpt_tp + gpt_fn) != 0 else 0
    gpt_f1_score = calculate_f1_score(gpt_tp, gpt_tn, gpt_fp, gpt_fn)
    
    fs_precision = fs_tp / (fs_tp + fs_fp) if (fs_tp + fs_fp) != 0 else 0
    fs_recall = fs_tp / (fs_tp + fs_fn) if (fs_tp + fs_fn) != 0 else 0
    fs_f1_score = calculate_f1_score(fs_tp, fs_tn, fs_fp, fs_fn)
    return {
            "gpt-4o-wiki_answer": {
                "TP": gpt_tp, "TN": gpt_tn, "FP": gpt_fp, "FN": gpt_fn,
                "Precision": gpt_precision, "Recall": gpt_recall,
                "F1_score": gpt_f1_score
            },
            "FS_answer": {
                "TP": fs_tp, "TN": fs_tn, "FP": fs_fp, "FN": fs_fn,
                "Precision": fs_precision, "Recall": fs_recall,
                "F1_score": fs_f1_score
            },
            "human_answer": {
                "all_counts": times
            },
            "gpt-fs_answer": {
                "TP": gpt_fs_tp, "TN": gpt_fs_tn
            },
            "gpt-fs-human_answer": {
                "TP": gpt_fs_human_tp, "TN": gpt_fs_human_tn
            }
        }
    


def plot_ven(metrics_dict):
    '''
    绘制维恩图
    '''
    # 定义每个集合的大小以及交集的大小
    counts = metrics_dict["human_answer"]["all_counts"]
    human_gpt_counts = metrics_dict["gpt-4o-wiki_answer"]["TP"] + metrics_dict["gpt-4o-wiki_answer"]["TN"]
    human_fs_counts = metrics_dict["FS_answer"]["TP"] + metrics_dict["FS_answer"]["TN"]
    fs_gpt_counts = metrics_dict["gpt-fs_answer"]["TP"] + metrics_dict["gpt-fs_answer"]["TN"]
    human_fs_gpt_counts = metrics_dict["gpt-fs-human_answer"]["TP"] + metrics_dict["gpt-fs-human_answer"]["TN"]

    # 数据定义
    # 数据示例
    set_sizes = {
        '100': 17,  # 只在A
        '010': 29,  # 只在B
        '001': 29,  # 只在C
        '110': 298,  # A和B
        '101': 286,   # A和C
        '011': 286,   # B和C
        '111': 398   # A, B和C
    }




    # set_labels = ("Human", "GPT-4o", "MuSKGC-FS")
    # set_sizes = {
    #     "100": counts-human_gpt_counts-human_fs_counts+human_fs_gpt_counts,  # 仅Human
    #     "010": counts-human_gpt_counts-fs_gpt_counts+human_fs_gpt_counts,  # 仅GPT
    #     "001": counts-human_fs_counts-fs_gpt_counts+human_fs_gpt_counts,  # 仅MuSKGC
    #     "110": human_gpt_counts-human_fs_gpt_counts,  # Human ∩ GPT
    #     "101": human_fs_counts-human_fs_gpt_counts,  # Human ∩ MuSKGC
    #     "011": fs_gpt_counts-human_fs_gpt_counts,  # GPT ∩ MuSKGC
    #     "111": human_fs_gpt_counts  # Human ∩ GPT ∩ MuSKGC
    # }
    total = sum(set_sizes.values())

    # 计算百分比并格式化
    def format_label(value):
        percentage = f"{value / total * 100:.1f}%"
        return f"{value} ({percentage})"

    # 创建维恩图
    plt.figure(figsize=(8, 6))#, dpi=1000
    venn = venn3(subsets=set_sizes, set_labels=None, set_colors=('#43b0f1','#b5ea8c',"#ff6361"), 
          alpha = 0.5)


    # 自定义颜色
    # circle_colors = ['#0073C2FF','#EFC000FF',"#CD534CFF"]  # 红、绿、蓝
    circle_colors = ['#43b0f1','#b5ea8c',"#ff6361"]
    

    for subset, value in set_sizes.items():
        if venn.get_label_by_id(subset):
            venn.get_label_by_id(subset).set_text(format_label(value))

            
    # 添加图例
    labels = ["Human", "GPT-4o", "MusKGC-FS"]  # 图例标签
    handles = [plt.Rectangle((0, 0), 2, 2, color=circle_colors[i], alpha=0.3) for i in range(3)]
    plt.legend(handles, labels, loc="upper center", fontsize=12, ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    plt.show()
    plt.savefig(f"small_experiments/select_query/ven_{args.task}.png", bbox_inches='tight', pad_inches=0)




if __name__ == "__main__":
    merge_dir = os.path.join(args.evaluation_dir, args.eval_type, "merge")
    files = os.listdir(merge_dir)
    metrics_dict = static_metrics(files)
    plot_ven(metrics_dict)

    print(metrics_dict)
