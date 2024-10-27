import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import logging
import json
import torch.nn as nn
import time
import torch
import numpy as np
import os
import tqdm
import math
import torch.nn.functional as F
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import re
triple_format = r'<(.*?)>'

with open("./entity_link/session_2/test_triple_context.json",'r',encoding='utf-8') as f:
    context_triple = json.load(f)

with open("./entity_link/session_2/test_triple_persona.json",'r',encoding='utf-8') as f:
    persona_triple = json.load(f)

with open("./entity_link/session_2/test_triple_query.json",'r',encoding='utf-8') as f:
    query_triple = json.load(f)

with open("./entity_link/session_2/peacok_linked_query_and_persona_test.json",'r',encoding='utf-8') as f:
    query_persona_kg_extend = json.load(f)

with open("./entity_link/session_2/peacok_linked_context_test.json",'r',encoding='utf-8') as f:
    context_kg_extend = json.load(f)

relations_dict = {
                      "here is my character trait": "characteristic",
                      "here is my character trait related to other people or social groups": "characteristic_relationship",
                      "here is what I regularly or consistently do related to other people or social groups" : "routine_habit_relationship",
                      "here is what I will do or achieve in the future related to other people or social groups" : "goal_plan_relationship",
                      "here is what I did in the past related to other people or social groups" : "experience_relationship",
    }

with open("./entity_link/session_4_pre/relation2id.json",'r',encoding='utf-8') as f:
    relation2id = json.load(f)

with open("./new_agent/session_2/test_data_with_context_and_memory.json",'r',encoding='utf-8') as f:
    dataset = json.load(f)
query_npy = []
node_npy = []
context_npy = []
response_npy = []
response2id = {}
query2id = {}
node2id = {}
context2id = {}


    
def get_train_data(data):
    query = data['query']
    context = data['context'] #可以更改context范围
    persona = data['init_personas'][1]
    graph = []
    
    for p in persona:
        if p in persona_triple.keys() and persona_triple[p] != 'None':
            p_t = re.findall(triple_format, persona_triple[p])
            for t in p_t:
                t = t.strip()
                t = t.split(", ")
                if len(t) != 3 :# 格式不对
                    continue
                if t[2] == 'None':
                    continue
                if t[1] not in relation2id.keys():
                    continue
                if t[0] == "Speaker" or t[0] == "I":
                    t[0] = "I"
                else:# 主语不对
                    continue
                graph.append(t)

                if p in query_persona_kg_extend.keys() and query_persona_kg_extend[p] != 'None':
                    extend_triple = query_persona_kg_extend[p]
                    for extend_t in extend_triple:
                        extend_t = extend_t.split(",")
                        if t[2] in extend_t[0]:
                            graph.append([t[2], relations_dict[extend_t[1]], extend_t[2]])

    for session in context:
        for utterance in session:
            if utterance in context_triple.keys() and context_triple[utterance] != 'None':
                c_t = re.findall(triple_format, context_triple[utterance])
                for t in c_t:
                    t = t.strip()
                    t = t.split(", ")
                    if len(t) != 3 :# 格式不对
                        continue
                    if t[2] == 'None':
                        continue
                    if t[1] not in relation2id.keys():
                        continue
                    if t[0] == "Speaker" or t[0] == "I":
                        t[0] = "I"
                    else:# 主语不对
                        continue
                    graph.append(t)

                    if utterance in context_kg_extend.keys() and context_kg_extend[utterance] != 'None':
                        extend_triple = context_kg_extend[utterance]
                        for extend_t in extend_triple:
                            extend_t = extend_t.split(",")
                            if t[2] in extend_t[0]:
                                graph.append([t[2], relations_dict[extend_t[1]], extend_t[2]])
                
    node_dict = {}
    edge_index = [[],[]]
    edge_type = []
    for triple in graph:
        if triple[0] not in node_dict:
            node_dict[triple[0]] = len(node_dict)
        if triple[2] not in node_dict:
            node_dict[triple[2]] = len(node_dict)
        #edge_index_user.append([user_node_dict[triple[0]], user_node_dict[triple[2]]])
        edge_index[0].append(node_dict[triple[0]])
        edge_index[1].append(node_dict[triple[2]])
        edge_type.append([relation2id[triple[1]]])

    # return dict
    return {
        "node_dict": list(node_dict.keys()),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "query": data['query'],
        "response": data['response'],
        "init_personas": data['init_personas'],
        "chabot_persona": data['chabot_persona'],
        "user_persona": data['user_persona'],
        #"memory": data['memory'],
        "relevant_context": data['relevant_context']
    }



class embed(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("./chatglm3-6b", trust_remote_code=True, device_map = 'auto')
        #self.device = "cuda:3"
        self.model_tokenizer = AutoTokenizer.from_pretrained("./chatglm3-6b", trust_remote_code=True)

    def forward(self, data, key = None):
        node = self.model_tokenizer(data, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.transformer(**node, output_hidden_states = True)
            embedding = list(outputs.hidden_states)
            last_hidden_states = embedding[-1].cpu().numpy()
            first_hidden_states = embedding[1].cpu().numpy()
            last_hidden_states = np.squeeze(last_hidden_states)
            first_hidden_states = np.squeeze(first_hidden_states)
            fisrt_larst_avg_status = np.mean(first_hidden_states + last_hidden_states, axis=0)
        return fisrt_larst_avg_status
        """if key == "relevant_context":
            query_driven = " ".join(data[key])
        else:
            query_driven = data[key]
        query_tokenzier = self.model_tokenizer(query_driven, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.transformer(**query_tokenzier, output_hidden_states = True)
            embedding = list(outputs.hidden_states)
            last_hidden_states = embedding[-1].cpu().numpy()
            first_hidden_states = embedding[1].cpu().numpy()
            last_hidden_states = np.squeeze(last_hidden_states)
            first_hidden_states = np.squeeze(first_hidden_states)
            fisrt_larst_avg_status = np.mean(first_hidden_states + last_hidden_states, axis=0)
        return fisrt_larst_avg_status"""

def main():
    with open("./new_agent/session_2/test_data_with_context_and_memory.json",'r',encoding='utf-8') as f:
        dataset = json.load(f)

    
    model = embed()
    for name,p in model.named_parameters():
        p.requires_grad = False

    for data in tqdm.tqdm(dataset):
        data = get_train_data(data)
        node_dict = data['node_dict']
        x = []
        for node in node_dict:
            if node not in node2id.keys():
                node2id[node] = len(node2id)
                if node == "I":
                    embedding = np.zeros((4096), dtype=float)
                else:
                    embedding = model(node)
                node_npy.append(embedding)
    with open("./vector_library/session_2/triple_from_doubao/node2id_test.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(node2id, indent=4, ensure_ascii=False))
    np.save("./vector_library/session_2/triple_from_doubao/node_test.npy", node_npy)

    """query = data['query']
        if query not in query2id.keys():
            query2id[query] = len(query2id)
            embedding = model(data, 'query')
            query_npy.append(embedding)
        context = " ".join(data['relevant_context'])
        if context not in context2id.keys():
            context2id[context] = len(context2id)
            embedding = model(data, "relevant_context")
            context_npy.append(embedding)
    with open("./vector_library/data_llama/query2id_train.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(query2id, indent=4, ensure_ascii=False))
    with open("./vector_library/data_llama/context2id_train.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(context2id, indent=4, ensure_ascii=False))
    np.save('./vector_library/data_llama/query_driven_train.npy',query_npy)
    np.save('./vector_library/data_llama/context_npy_train.npy',context_npy)"""
    """response = data['response']
        if response not in response2id.keys():
            response2id[response] = len(response2id)
            embedding = model(data, 'response')
            response_npy.append(embedding)
    with open("./vector_library/response2id_train.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(response2id, indent=4, ensure_ascii=False))
    np.save('./vector_library/response_train.npy',response_npy)"""



if __name__ == "__main__":
    main()
