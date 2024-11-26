
import math
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_simlarity(query,neigh_list,device):
    
    # 加载预训练的模型 sx修改预训练语言模型的路径到本地的models
    model = SentenceTransformer('/home/hyy/models/bert-base-nli-mean-tokens')

    # 定义句子
    sentences_A = query
    sentences_B = neigh_list

    # 使用模型为句子编码
    embeddings_A = model.encode(sentences_A)
    embeddings_B = model.encode(sentences_B)

    # 计算句子之间的余弦相似度
    similarity_matrix = cosine_similarity(embeddings_A, embeddings_B)[0]
    return similarity_matrix   #是个array


'''
 输入参数：
 q：一句话，用了关系模板之后转换的句子；
 neighbor_list=['n1','n2','',''],
 每个元素 ni 表示邻居三元组利用关系模板之后转换的句子.
 
 '''   
def get_score_q_neighbors(q,neighbor_lis):
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    score_list=get_simlarity(q, neighbor_lis,device).tolist()
    # 使用zip将两个列表组合，并根据得分排序
    sorted_pairs = sorted(zip(neighbor_lis, score_list), key=lambda pair: pair[1], reverse=True)
    # 提取排序后的列表
    sorted_names = [name for name, score in sorted_pairs]

    return sorted_names
   

    
if __name__=='__main__':
    #  调用案例demo
    # q="What is Robert Nozick interested in ?"
    q=["What is Robert Nozick interested in ?"]
    neighbors_list=["Robert Nozick influences Anita L. Allen",
                    "Robert Nozick died in Cambridge, Massachusetts",
                    "Robert Nozick has won the National Book Award"
                    ]
    

    sorted_nei_list=get_score_q_neighbors(q,neighbors_list)  #
    print(sorted_nei_list)
    print('sort finish')