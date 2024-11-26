import torch
import os
import re
import json

from logger import logger
from data_preprocess.doc import MyDataset, collate, get_neighbor_desc
# from wiki.wiki_search import search_wiki_doc
from llm.llm import MyLLM
from wiki.bge_reranker import calcu_score
from sx_template_type.sample_neighbor_by_sentenceBert import get_score_q_neighbors

class Trainer:
    def __init__(self, args):
        self.args = args

        
        self.client = MyLLM()
        self.relation_data, self.relation_template  = self.load_relation_data()
        self.nei_limitation = self.args.nei_limitation
        
    
    def load_mydataset(self, path):
        """
        
        """
        logger.info("=> Creating data loader...")
        test_dataset = MyDataset(path)
        self.test_path = path
        self.info = list()
        self.retrive_data = dict()
        self.llm_answer = dict()
        
        self.train_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=False
        )

        self.file = self.test_path.split('/')[-1]
        self.save_info_path = self.test_path.split('.json')[0] + "_info.json"
        self.retriver_result_path = os.path.join(self.args.retriver_dir, self.file)
        self.LLM_result_path = os.path.join(self.args.LLM_result_dir, self.file)
        

    def load_json_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def load_relation_data(self):
        data = self.load_json_data(self.args.relation_path)
        relation_data = {v: k for k,v in data.items()}

        relation_template = self.load_json_data(self.args.relation_template_path)

        return relation_data, relation_template

    def store_json_file(self, data, path):
        """
        data: list[{},{},{}]
        path: 存储路径
        """
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def neighbors_template2text(self, neighbors):
        """
        neighbors: list()   # ['genus bombax(head entity)\tinverse member meronym(relation)\tfamily bombacaceae(tail entity)', 'genus bombax\thypernym\tdilleniid dicot genus']
        """
        if self.args.task == "WN18RR":
            pass

        text_list = list()
        for item in neighbors:
            h, r, t = item.split('\t')
            r_template = self.relation2template(r)  # "The county seat of [X] is [Y] ."
            text = r_template.replace('[X]', h).replace('[Y]', t)
            text_list.append(text)
        return text_list


    def load_from_kg(self, head_id, relation, query, limitation):
        """
        给定三元组(h,r,?)或(?,,r,t)找出h的邻居(top limitation个)(可以参考triplets.py中的get_n_hop_entity_indices实现)
        """
        neighbors = get_neighbor_desc(head_id, relation, self.args.save_neighbors_path, filter_relation=True) # ['genus bombax(head entity)\tinverse member meronym(relation)\tfamily bombacaceae(tail entity)', 'genus bombax\thypernym\tdilleniid dicot genus']   head_id的邻居信息，除去相同<h,r>的邻居
        # text_neighbors = self.neighbors_template2text(neighbors)
        try:
            # sorted_text_neighbors = get_score_q_neighbors([query], text_neighbors)[:limitation]   #list()   #!有可能存在text_neighbors为[]的情况
            sorted_neighbors = get_score_q_neighbors([query], neighbors)[:limitation]
        except Exception:
            # return text_neighbors, neighbors
            return neighbors

        # return sorted_text_neighbors, sorted_neighbors
        return sorted_neighbors

    def relation2template(self, relation):
        """
        根据关系检索关系模板
        1.首先根据relations.json转变为带符号/的关系格式
        2.根据上一步转变后的关系，到relation2template检索关系模板
        """
        return self.relation_template[self.relation_data[relation]].strip() #str
    
    def triplet2Query(self, head_name, relation, tail_name):
        f_query, b_query = None, None
        try:
            relation_template = self.relation2template(relation)

            ##(h,r,?)
            forward_relation_template = relation_template.replace('[X]', head_name).replace('[Y]', "[MASK]")
            query_triplet = "Please transform the following sentence into a question asking [MASK]: {r_template}.".format(r_template=forward_relation_template)
            f_query = self.client.triplet2Query(query_triplet, model="gpt-4o-mini")
            # f_query_2 = self.client.triplet2Query(query_triplet)

            #判断[MASK]
            if "[mask]" in f_query.lower():
                f_query = self.client.triplet2Query(query_triplet, model="gpt-4o-mini")
            ##(h,r,?)

            ##(?,r,t)
            backward_relation_template = relation_template.replace('[X]', '[MASK]').replace('[Y]', tail_name)
            q_triplet = "Please transform the following sentence into a question asking [MASK]: {r_template}.".format(r_template=backward_relation_template)
            b_query = self.client.triplet2Query(q_triplet, model="gpt-4o-mini")
            # b_query_2 = self.client.triplet2Query(q_triplet)
            #判断[MASK]
            if "[mask]" in b_query.lower():
                b_query = self.client.triplet2Query(query_triplet, model="gpt-4o-mini")
        except Exception:
            print(head_name, relation, tail_name)
        return f_query, b_query

    def load_from_wiki(self, f_query, b_query, head_name, tail_name, head_description, tail_description, relation):
        '''
        根据query从wiki库里面检索知识
        '''
        ##(h,r,?)
        head_name = ' '.join(head_name.split('_')[:-2])
        tail_name = ' '.join(tail_name.split('_')[:-2])

        query_triplet_forward = f_query + " <{h}>: {h_desc}.".format(h=head_name, h_desc=head_description)
        ##(h,r,?)

        ##(?,r,t)
        query_triplet_backward = b_query + " <{t}>: {t_desc}.".format(t=tail_name, t_desc=tail_description)
        ##(?,r,t)

        query = [query_triplet_forward, query_triplet_backward]
        retrive_docs = search_wiki_doc(query, 10)   # list[[]]

        #BEG reranker
        top_forward_doc = calcu_score(query_triplet_forward, retrive_docs[0], top_k=True)
        top_backward_doc = calcu_score(query_triplet_backward, retrive_docs[1], top_k=True)
        #BEG reranker

        # 存储
        if head_name+"||"+relation not in self.retrive_data:
            self.retrive_data[head_name+"||"+relation] = list()
        self.retrive_data[head_name+"||"+relation].append({"tail": tail_name, "query": query_triplet_forward, "retrive_docs": top_forward_doc})
        if tail_name+"||inverse "+relation not in self.retrive_data:
            self.retrive_data[tail_name+"||inverse "+relation] = list()
        self.retrive_data[tail_name+"||inverse "+relation].append({"head": head_name, "query": query_triplet_backward, "retrive_docs": top_backward_doc})
        # 存储

        return top_forward_doc, top_backward_doc

    def process_llm_answer(self, answer):
        answer_list = []
        if '[' in answer and ']' in answer:
            try:
                result = re.findall(r'\[(.*?)\]', answer)
                answer_list = [answer.strip() for answer in result[0].split(',')]
            except Exception:
                answer_list = [answer]
        else:
            answer_list = [answer]
        return answer_list
    

    def generateQuery(self):
        for file in os.listdir(self.args.test_dir):
            path = os.path.join(self.args.test_dir, file)
            with open(path) as f:
                test_data = json.load(f)

            prefix = file.split('.json')[0]
            save_path = os.path.join(self.args.test_dir, prefix+"_gpt4omini.json")
            new_test = list()
            
            for item in test_data:
                relation = item["relation"]
                head_name = item["head"]
                tail_name = item["tail"]

                ##(h,r,?)
                f_query, b_query = self.triplet2Query(head_name, relation, tail_name)
                ##(?,r,t)

                item["query_t"] = f_query
                item["query_h"] = b_query
                new_test.append(item)

            self.store_json_file(new_test, save_path)   #存储每个三元组生成的query文件
        

    def decouple_data(self, data, use_desc=True):
        head_id = data.head_id
        head = data.head
        relation = data.relation
        tail_id = data.tail_id
        tail = data.tail
        query_t = data.query_t
        query_h = data.query_h
        head_description = data.head_desc
        tail_description = data.tail_desc

        if use_desc:
            return head_id, head, relation, tail_id, tail, query_t, query_h, head_description, tail_description
        
        return head_id, head, relation, tail_id, tail, query_t, query_h

    def save_information(self, data, sorted_text_neighbors_forward, sorted_neighbors_forward, sorted_text_neighbors_backward, sorted_neighbors_backward, top_forward_doc, top_backward_doc, forward_answer_list, backward_answer_list, store_nei=True, store_wiki=True, store_llm=True):
        """
        将从KG中检索的邻居信息(排序后的前limitation个)和wiki检索的结果(top 1)和原本text.json中的数据存储在一起

        data : batch
        sorted_text_neighbors_forward : list()
        sorted_text_neighbors_backward : list()
        top_forward_doc : list()
        top_backward_doc : list()
        """
        item = dict()
        head_id, head, relation, tail_id, tail, query_t, query_h = self.decouple_data(data, use_desc=False)
        item["head_id"] = head_id
        item["head"] = head
        item["relation"] = relation
        item["tail_id"] = tail_id
        item["tail"] = tail
        item["query_t"] = query_t
        item["query_h"] = query_h
        if store_nei:
            item["h_neighbors"] = sorted_neighbors_forward
            item["h_neighbors_info"] = sorted_text_neighbors_forward
            item["t_neighbors"] = sorted_neighbors_backward
            item["t_neighbors_info"] = sorted_text_neighbors_backward
        if store_wiki:
            item["forward_wiki_info"] = top_forward_doc[:self.args.wiki_limitation]
            item["backward_wiki_info"] = top_backward_doc[:self.args.wiki_limitation]
        if store_llm:
            item["forward_answer"] = forward_answer_list
            item["backward_answer"] = backward_answer_list

        self.info.append(item)

    def get_neighbor(self, path):
        data = self.load_json_data(path)
        l = list()
        for item in data:
            head_id = item["head_id"]
            tail_id = item["tail_id"]
            relation = item["relation"]
            query_t = item["query_t"]
            query_h = item["query_h"]
            neighbors_forward = self.load_from_kg(head_id, relation, query_t, self.nei_limitation)
            neighbors_backward = self.load_from_kg(tail_id, relation, query_h, self.nei_limitation)
            item["h_neighbors"] = neighbors_forward
            item["t_neighbors"] = neighbors_backward
            l.append(item)
        self.store_json_file(l, path)

    def train(self):
        begin, end = 141 , 205
        files = ["test_" + str(i) + "_gpt4omini_allinfo.json" for i in range(begin, end+1)]
        files_path = [os.path.join(self.args.test_dir, f) for f in files]
        for test_path in files_path:
            self.load_mydataset(test_path)
            self.get_neighbor(test_path)
            '''
            # for i, batch_dict in enumerate(self.train_loader):
            #     batch_data = batch_dict['batch_data']

                # cnt = len(batch_data)
                # for i in range(cnt):
                #     data = batch_data[i]
                #     head_id, head_name, relation, tail_id, tail_name, query_t, query_h, head_description, tail_description = self.decouple_data(data)

                    
                    # sorted_text_neighbors_forward, sorted_neighbors_forward = self.load_from_kg(head_id, relation, query_t, self.nei_limitation)
                    # sorted_text_neighbors_backward, sorted_neighbors_backward = self.load_from_kg(tail_id, relation, query_h, self.nei_limitation) #list()

                    
                    top_forward_doc, top_backward_doc = self.load_from_wiki(query_t, query_h, head_name, tail_name, head_description, tail_description, relation)
                    
                    forward_answer_list = []
                    backward_answer_list = []
                    self.save_information(data, sorted_text_neighbors_forward, sorted_neighbors_forward, sorted_text_neighbors_backward, sorted_neighbors_backward, top_forward_doc, top_backward_doc, forward_answer_list, backward_answer_list, store_llm=False)


            self.store_json_file(self.retrive_data, self.retriver_result_path)  #存储wiki检索的文档
            # self.store_json_file(self.llm_answer, self.LLM_result_path)    #存储LLM最终回答的文档
            self.store_json_file(self.info, self.save_info_path)    #存储从KG和Wiki中检索出的信息
            '''

            
                    