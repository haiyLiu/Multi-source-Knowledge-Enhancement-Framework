import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from logger import logger
from config import args
from data_preprocess.dict_hub import get_entity_dict, get_link_graph
from data_preprocess.triplet import reverse_triplet

entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, r: str, save_neighbors_path, filter_relation=False) -> List:
    neighbors = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    neighbors_list = []
    with open(save_neighbors_path, "a+") as f:
        head_entity = _parse_entity_name(entity_dict.get_entity_by_id(head_id).entity)
        try:
            for relation, neighbor_ids_set in neighbors.items():    #!可能会出现找不到邻居节点的情况
                neighbor_ids = [n_id for n_id in neighbor_ids_set]  ## neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
                tail_entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
                tail_entities = [_parse_entity_name(entity) for entity in tail_entities]
                if filter_relation:
                    if relation.strip().lower() == r.strip().lower():
                        continue
                for tail_entity in tail_entities:
                    triplet = head_entity + '\t' + relation + '\t' + tail_entity
                    neighbors_list.append(triplet)
                    f.write(triplet+'\n')
        except Exception:
            print("***********ERROR*************")
            print(head_id, head_entity)
            pass
    # entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    # entities = [_parse_entity_name(entity) for entity in entities]
    return neighbors_list


class Example:

    def __init__(self, head_id, relation, tail_id, query_t, query_h, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        self.query_t = query_t
        self.query_h = query_h

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        # if args.use_link_graph:
        #     if len(head_desc.split()) < 20:
        #         head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
        #     if len(tail_desc.split()) < 20:
        #         tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        # hr_encoded_inputs = _custom_tokenize(text=head_text,
                                            #  text_pair=self.relation)

        # head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_text = _concat_name_desc(tail_word, tail_desc)
        # tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        return {
             'head_id': self.head_id, 'head_word': head_word, 'head_desc': head_desc, 'head_text': head_text,
             'tail_id': self.tail_id, 'tail_word': tail_word, 'tail_desc': tail_desc, 'tail_text': tail_text,
             'relation': self.relation,
             'query_t': self.query_t, 'query_h': self.query_h,
             'obj': self

        }

        # return {'hr_token_ids': hr_encoded_inputs['input_ids'],
        #         'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
        #         'tail_token_ids': tail_encoded_inputs['input_ids'],
        #         'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
        #         'head_token_ids': head_encoded_inputs['input_ids'],
        #         'head_token_type_ids': head_encoded_inputs['token_type_ids'],
        #         'obj': self}


class MyDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, examples=None):
        self.path_list = path.split(',')
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = False) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    head_id = [ex['head_id'] for ex in batch_data]
    head_word = [ex['head_word'] for ex in batch_data]
    head_desc = [ex['head_desc'] for ex in batch_data]
    head_text = [ex['head_text'] for ex in batch_data]

    tail_id = [ex['tail_id'] for ex in batch_data]
    tail_word = [ex['tail_word'] for ex in batch_data]
    tail_desc = [ex['tail_desc'] for ex in batch_data]
    relation = [ex['relation'] for ex in batch_data]
    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'head_id': head_id,
        'head_word': head_word,
        'head_desc': head_desc,
        'head_text': head_text,
        'tail_id': tail_id,
        'tail_word': tail_word,
        'tail_desc': tail_desc,
        'relation': relation,
        'batch_data': batch_exs
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
