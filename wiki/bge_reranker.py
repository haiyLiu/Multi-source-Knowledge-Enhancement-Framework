from FlagEmbedding import FlagReranker
import torch
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from config import args

reranker = FlagReranker(args.bge_model, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# reranker = FlagReranker("/home/kayla/models/bge-reranker-v2-m3", use_fp16=True)

def calcu_score(query, retrieval_doc, top_k=False):
    """
    query: str
    retrieval_doc: list[]
    top: bool
    """
    all = list()
    for doc in retrieval_doc:
        l = [query, doc["text"]]
        all.append(l)
    # print(all)
    scores = reranker.compute_score(all, normalize=True)
    if top_k:
        top_doc = select_topk(retrieval_doc, scores)
        return top_doc  #list[]
    return scores   #list[]

def select_topk(retrieval_doc, scores, k=5):
    """
    retrieval_doc: list[]
    scores: list[]
    k: int
    """
    scores_arr = np.array(scores)
    topK_indices = np.argsort(scores_arr)[-k:][::-1]  # 获取前K个最大值的索引
    topK_values = scores_arr[topK_indices]  # 获取前K个最大值
    top_doc = [retrieval_doc[i] for i in topK_indices]
    # print("Top K 值:", topK_values)
    # print("Top K 索引:", topK_indices)
    return top_doc


# retrieval_doc = ["Robert Nozick has won the National Book Award.", "Robert Nozick influences Anita L. Allen.", "Robert Nozick died in Cambridge, Massachusetts."]
# query = "What is Robert Nozick interested in?"
# scores = calcu_score(query, retrieval_doc)
# top_doc = select_topk(retrieval_doc, scores)
# print(scores)
# print(top_doc)


# scores=reranker.compute_score([['What is Robert Nozick interested in?','Robert Nozick has won the National Book Award.'],
#                                ['What is Robert Nozick interested in?','Robert Nozick influences Anita L. Allen.'],
#                                ['What is Robert Nozick interested in?','Robert Nozick died in Cambridge, Massachusetts.']],normalize=True )
# print(scores)