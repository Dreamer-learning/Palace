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
query_npy = []
node_npy = []
context_npy = []
response_npy = []
response2id = {}
query2id = {}
node2id = {}
context2id = {}

def get_relations_dict():
    relations_dict = {"persona_explicited": 0, 
                      "persona_implicited": 1,
                      "characteristic": 2,
                      "characteristic_relationship": 3,
                      #"routine_habit": 4,
                      "routine_habit_relationship": 4,
                      #"goal_plan": 6,
                      "goal_plan_relationship": 5,
                      "experience_relationship": 6,
                      #"experience": 9
                      }
    return relations_dict

def filter_kg(persona_kg, relations_dict):
    persona_kg_filted = {}
    for key in persona_kg:
        persona_kg_filted[key] = []
        for triple in persona_kg[key]:
            if triple[0] in relations_dict.keys():
                persona_kg_filted[key].append(triple)

    return persona_kg_filted

def get_entity_link():
    entity_link = {}
    with open("./dialogue_kg/entity_matching_persona_history_with_summary.json",'r',encoding='utf-8') as f:
        entity_context = json.load(f)

    with open("./dialogue_kg/entity_matching_persona_history_with_summary_last_session.json",'r',encoding='utf-8') as f:
        entity_context_last_summary = json.load(f)

    with open("./dialogue_kg/entity_matching.json",'r',encoding='utf-8') as f:
        entity_persona = json.load(f)
    
    for key, value in entity_context.items():
        if key not in entity_link.keys():
            entity_link[key] = value
    
    for key, value in entity_context_last_summary.items():
        if key not in entity_link.keys():
            entity_link[key] = value
        
    for key, value in entity_persona.items():
        if key not in entity_link.keys():
            entity_link[key] = value
    return entity_link


def get_entity_link_valid():
    with open("./dialogue_kg/entity_matching_valid.json",'r',encoding='utf-8') as f:
        entity_context = json.load(f)
    return entity_context


def get_entity_link_test():
    with open("./dialogue_kg/entity_matching_test.json",'r',encoding='utf-8') as f:
        entity_context = json.load(f)
    return entity_context

    
def get_train_data(data, entity_link, persona_kg, node_dict, relations_dict):
    persona = data['init_personas']
    user_persona = persona[0]
    chatbot_persona = persona[1]
    context = data['context']
    user_context = []
    chatbot_context = []
    for session in context:
        if len(session) == 0:
            break
        for i in range(len(session)):
            if i % 2 == 0:
                user_context.append(session[i])
            else:
                chatbot_context.append(session[i])
    
    # for user graph
    user_graph = []
    s_dict_user = {}
    s_dict_user['User'] = [1,0,0]
    explicted_linked_user = []
    for persona in user_persona:
        if persona in entity_link.keys():
            if entity_link[persona] != 'None' and entity_link[persona] not in explicted_linked_user:
                user_graph.append(["User", "persona_explicited", entity_link[persona]])
                explicted_linked_user.append(entity_link[persona])
                s_dict_user[entity_link[persona]] = [0,1,0]
                for tail in persona_kg[entity_link[persona]]:
                    user_graph.append([entity_link[persona], tail[0], tail[1]])
                    s_dict_user[tail[1]] = [0,1,0]
            else:
                user_graph.append(["User","persona_explicited",persona])
                s_dict_user[persona] = [0,1,0]
        else:
            user_graph.append(["User","persona_explicited",persona])
            s_dict_user[persona] = [0,1,0]
    
    for context in user_context:
        if context in entity_link.keys():# 链接不准，可能出现同样的链接节点
            if entity_link[context] != 'None' and entity_link[context] not in explicted_linked_user:
                user_graph.append(["User", "persona_implicited", entity_link[context]])
                s_dict_user[entity_link[context]] = [0,0,1]
                if entity_link[context] in persona_kg.keys():
                    for tail in persona_kg[entity_link[context]]:
                        user_graph.append([entity_link[context], tail[0], tail[1]])
                        s_dict_user[tail[1]] = [0,0,1]
                        

    user_node_dict = {}
    edge_index_user = [[],[]]
    edge_type_user = []
    for triple in user_graph:
        if triple[0] not in user_node_dict:
            user_node_dict[triple[0]] = len(user_node_dict)
        if triple[2] not in user_node_dict:
            user_node_dict[triple[2]] = len(user_node_dict)
        #edge_index_user.append([user_node_dict[triple[0]], user_node_dict[triple[2]]])
        edge_index_user[0].append(user_node_dict[triple[0]])
        edge_index_user[1].append(user_node_dict[triple[2]])
        edge_type_user.append([relations_dict[triple[1]]])
    adj_user = [[0] * len(user_node_dict) for i in range(len(user_node_dict))]
    s_user = []
    for edge in edge_index_user:
        adj_user[edge[0]][edge[1]] = 1
    for node in user_node_dict:
        s_user.append(s_dict_user[node])


    chatbot_graph = []
    s_dict_chatbot = {}
    s_dict_chatbot['Chatbot'] = [1,0,0]
    explicted_linked_chatbot = []
    for persona in chatbot_persona:
        if persona in entity_link.keys():
            if entity_link[persona] != 'None':
                chatbot_graph.append(["Chatbot", "persona_explicited", entity_link[persona]])
                explicted_linked_chatbot.append(entity_link[persona])
                s_dict_chatbot[entity_link[persona]] = [0,1,0]
                for tail in persona_kg[entity_link[persona]]:
                    chatbot_graph.append([entity_link[persona], tail[0], tail[1]])
                    s_dict_chatbot[tail[1]] = [0,1,0]
            else:
                chatbot_graph.append(["Chatbot","persona_explicited",persona])
                s_dict_chatbot[persona] = [0,1,0]
            
        else:
            chatbot_graph.append(["Chatbot","persona_explicited",persona])
            s_dict_chatbot[persona] = [0,1,0]
            
    for context in chatbot_context:
        if context in entity_link.keys():
            if entity_link[context] != 'None' and entity_link[context] not in explicted_linked_chatbot:
                chatbot_graph.append(["Chatbot", "persona_implicited", entity_link[context]])
                s_dict_chatbot[entity_link[context]] = [0,0,1]
                if entity_link[context] in persona_kg.keys():
                    for tail in persona_kg[entity_link[context]]:
                        chatbot_graph.append([entity_link[context], tail[0], tail[1]])
                        s_dict_chatbot[tail[1]] = [0,0,1]
    
    chatbot_node_dict = {}
    edge_index_chatbot = [[],[]]
    edge_type_chatbot = []
    for triple in chatbot_graph:
        if triple[0] not in chatbot_node_dict:
            chatbot_node_dict[triple[0]] = len(chatbot_node_dict)
        if triple[2] not in chatbot_node_dict:
            chatbot_node_dict[triple[2]] = len(chatbot_node_dict)
        #edge_index_chatbot.append([chatbot_node_dict[triple[0]], chatbot_node_dict[triple[2]]])
        edge_index_chatbot[0].append(chatbot_node_dict[triple[0]])
        edge_index_chatbot[1].append(chatbot_node_dict[triple[2]])
        edge_type_chatbot.append([relations_dict[triple[1]]])
    
    adj_chatbot = [[0] * len(chatbot_node_dict) for i in range(len(chatbot_node_dict))]
    for edge in edge_index_chatbot:
        adj_chatbot[edge[0]][edge[1]] = 1
    s_chatbot = []
    for node in chatbot_node_dict:
        s_chatbot.append(s_dict_chatbot[node])

    # return dict
    return {
        "user_node_dict": list(user_node_dict.keys()),
        "edge_index_user": edge_index_user,
        "edge_type_user": edge_type_user,
        "adj_user": adj_user,
        "s_user":s_user,
        "chatbot_node_dict": list(chatbot_node_dict.keys()),
        "edge_index_chatbot": edge_index_chatbot,
        "edge_type_chatbot": edge_type_chatbot,
        "adj_chatbot": adj_chatbot,
        "s_chatbot": s_chatbot,
        "query": data['query'],
        "response": data['response'],
        "init_personas": data['init_personas'],
        "chabot_persona": data['chabot_persona'],
        "user_persona": data['user_persona'],
        "memory": data['memory'],
        "relevant_context": data['relevant_context']
    }



class embed(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("/home/data/public/llama2", trust_remote_code=True, device_map = 'auto')
        #self.model = AutoModelForCausalLM.from_pretrained("./chatglm3-6b", trust_remote_code=True, device_map = 'auto')
        #self.device = "cuda:3"
        self.model_tokenizer = AutoTokenizer.from_pretrained("/home/data/public/llama2", trust_remote_code=True)

    def forward(self, data, key = None):
        """node = self.model_tokenizer(data, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.model(**node, output_hidden_states = True)
            embedding = list(outputs.hidden_states)
            last_hidden_states = embedding[-1].cpu().numpy()
            first_hidden_states = embedding[1].cpu().numpy()
            last_hidden_states = np.squeeze(last_hidden_states)
            first_hidden_states = np.squeeze(first_hidden_states)
            fisrt_larst_avg_status = np.mean(first_hidden_states + last_hidden_states, axis=0)
        return fisrt_larst_avg_status"""
        if key == "relevant_context":
            query_driven = " ".join(data[key])
        else:
            query_driven = data[key]
        query_tokenzier = self.model_tokenizer(query_driven, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.model(**query_tokenzier, output_hidden_states = True)
            embedding = list(outputs.hidden_states)
            last_hidden_states = embedding[-1].cpu().numpy()
            first_hidden_states = embedding[1].cpu().numpy()
            last_hidden_states = np.squeeze(last_hidden_states)
            first_hidden_states = np.squeeze(first_hidden_states)
            fisrt_larst_avg_status = np.mean(first_hidden_states + last_hidden_states, axis=0)
        return fisrt_larst_avg_status

def main():
    with open("./new_agent/session_4/test_data_with_context_and_memory.json",'r',encoding='utf-8') as f:
        dataset = json.load(f)
    with open("./dialogue_kg/peacok_triple_dict.json",'r',encoding='utf-8') as f:
        persona_kg = json.load(f)
    entity_link = get_entity_link()
    relations_dict = get_relations_dict()
    persona_kg = filter_kg(persona_kg, relations_dict)    
    
    model = embed()
    for name,p in model.named_parameters():
        p.requires_grad = False

    for data in tqdm.tqdm(dataset):
        """data = get_train_data(data, entity_link, persona_kg, node_dict, relations_dict)
        chatbot_node_dict = data['chatbot_node_dict']
        chatbot_x = []
        for node in chatbot_node_dict:
            if node == 'Chatbot':
                node = " ".join(data['init_personas'][1])
            if node not in node2id.keys():
                node2id[node] = len(node2id)
                embedding = model(node)
                node_npy.append(embedding)
    with open("./vector_library/data_llama/node2id_train.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(node2id, indent=4, ensure_ascii=False))
    np.save("./vector_library/data_llama/node_train.npy", node_npy)"""

        query = data['query']
        if query not in query2id.keys():
            query2id[query] = len(query2id)
            embedding = model(data, 'query')
            query_npy.append(embedding)
        #context = " ".join(data['relevant_context'])
        #if context not in context2id.keys():
        #    context2id[context] = len(context2id)
        #    embedding = model(data, "relevant_context")
        #    context_npy.append(embedding)
    with open("./vector_library/session_4/data_llama/query2id_test.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(query2id, indent=4, ensure_ascii=False))
    #with open("./vector_library/data_llama/context2id_train.json",'w',encoding='utf-8') as f:
    #    f.write(json.dumps(context2id, indent=4, ensure_ascii=False))
    np.save('./vector_library/session_4/data_llama/query_driven_test.npy',query_npy)
    #np.save('./vector_library/data_llama/context_npy_train.npy',context_npy)
    """    response = data['response']
        if response not in response2id.keys():
            response2id[response] = len(response2id)
            embedding = model(data, 'response')
            response_npy.append(embedding)
    with open("./vector_library/response2id_train.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(response2id, indent=4, ensure_ascii=False))
    np.save('./vector_library/response_train.npy',response_npy)"""



if __name__ == "__main__":
    main()
