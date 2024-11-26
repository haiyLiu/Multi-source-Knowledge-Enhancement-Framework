import os
import sys 
from evaluate import load_entity_information, special_nei
sys.path.append("/home/kayla/lhy/code/multi-source") 
from base_methods import load_json_data, save_json_data, check_dir
from data_preprocess.triplet import LinkGraph

def filter_predicted_entities(hr_nei_name, predicted_entities):
    # 将 hr_nei_name 转换为集合以提高查找效率
    hr_nei_set = set(hr_nei_name)
    
    # 记录被过滤掉的实体
    filtered_out = [entity for entity in predicted_entities if entity in hr_nei_set]
    
    # 过滤掉包含 hr_nei_name 实体的 predicted_entities
    filtered_entities = [entity for entity in predicted_entities if entity not in hr_nei_set]
    
    return filtered_entities, filtered_out


if __name__ == "__main__":
    answer_dir = "data/FB15k237/llm_answers/gpt-4o-mini"
    filter_dir = "data/FB15k237/llm_answers/filter_labels"
    entity_path = "data/FB15k237/entities.json"
    link_graph = LinkGraph(train_path="data/FB15k237/test.txt.json", fact_path='')
    entity_id2name = load_entity_information(entity_path)

    for file in os.listdir(answer_dir):
        answer_data = load_json_data(os.path.join(answer_dir, file))
        new_list = list()
        for item in answer_data:
            forward_predicted_entities = item["forward_predicted_entities"]
            head_id = item["head_id"]
            relation = item["relation"]
            hr_nei_ids = special_nei(head_id, relation, link_graph, inverse=False)
            hr_nei_name = [entity_id2name[id] for id in hr_nei_ids]
            forward_filtered_entities, forward_filtered_out = filter_predicted_entities(hr_nei_name, forward_predicted_entities)


            backward_predicted_entities = item["backward_predicted_entities"]
            tail_id = item["tail_id"]
            backward_hr_nei_ids = special_nei(tail_id, relation, link_graph, inverse=True)
            backward_hr_nei_name = [entity_id2name[id] for id in backward_hr_nei_ids]
            backward_filtered_entities, backward_filtered_out = filter_predicted_entities(backward_hr_nei_name, forward_predicted_entities)

            del item["forward_prompt"]
            del item["forward_sorted_candidates"]
            del item["forward_predicted_entities"]
            del item["backward_prompt"]
            del item["backward_sorted_candidates"]
            del item["backward_predicted_entities"]

            item["forward_predicted_entities_outoflabels"] = forward_filtered_entities
            item["forward_filtered_out_entities"] = forward_filtered_out

            item["backward_predicted_entities_outoflabels"] = backward_filtered_entities
            item["backward_filtered_out_entities"] = backward_filtered_out

            new_list.append(item)

        save_json_data(os.path.join(filter_dir, file), new_list)