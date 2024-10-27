from transformers import AutoTokenizer
from plms.llama import LlamaForCausalLM 
from peft import LoraConfig, get_peft_model, TaskType
import logging
import json
import torch.nn as nn
import time
import torch
import numpy as np
import os
from tkinter import _flatten
import tqdm
import math
import torch.nn.functional as F
from peft.utils import transpose, _get_submodules
import warnings
from typing import Optional, List
import re
from vae import VAE
from rgcn import RGCN
import argparse
triple_format = r'<(.*?)>'
relations_dict = {
                      "here is my character trait": "characteristic",
                      "here is my character trait related to other people or social groups": "characteristic_relationship",
                      "here is what I regularly or consistently do related to other people or social groups" : "routine_habit_relationship",
                      "here is what I will do or achieve in the future related to other people or social groups" : "goal_plan_relationship",
                      "here is what I did in the past related to other people or social groups" : "experience_relationship",
    }

with open("./relation2id.json",'r',encoding='utf-8') as f:
    relation2id = json.load(f)

def save_checkpoint(model, optimizer, cur_epoch, save_path):
    param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
        if k[22:] in param_grad_dic.keys() and not param_grad_dic[k[22:]]:
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
    }
    save_to = os.path.join(
        save_path,
        "llama_ppl_{}".format(cur_epoch),
    )
    logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


    
def get_train_data(data, context_triple, persona_triple, query_triple, query_persona_kg_extend, context_kg_extend, session_num):
    query = data['query']
    context = data['context'] 
    persona = data['init_personas'][1]
    graph = []
    
    for p in persona:
        if p in persona_triple.keys() and persona_triple[p] != 'None':
            p_t = re.findall(triple_format, persona_triple[p])
            for t in p_t:
                t = t.strip()
                t = t.split(", ")
                if len(t) != 3 :
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
                    if len(t) != 3 :
                        continue
                    if t[2] == 'None':
                        continue
                    if t[1] not in relation2id.keys():
                        continue
                    if t[0] == "Speaker" or t[0] == "I":
                        t[0] = "I"
                    else:
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
        edge_index[0].append(node_dict[triple[2]])
        edge_index[1].append(node_dict[triple[0]])
        edge_type.append([relation2id[triple[1]]])

    # return dict
    return {
        "node_dict": list(node_dict.keys()),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "context": data['context'],
        "query": data['query'],
        "response": data['response'],
        "init_personas": data['init_personas'],
        #"chabot_persona": data['chabot_persona'],
        #"user_persona": data['user_persona'],
        "long_memory": data['long_memory'],
        "short_memory": data['short_memory']
    }


def _replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    if child_name == 'embedding':
        new_module.weight = old_module.word_embeddings.weight

        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
        for name, p in new_module.named_parameters():
            if "lora_" in name:
                p.to(old_module.word_embeddings.weight.device)
        
    else:
        new_module.weight = old_module.weight
        if getattr(old_module, "state", None) is not None:
            new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

            # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
        for name, p in new_module.named_parameters():
            if "lora_" in name:
                p.to(old_module.weight.device)

class PLoraLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_P_A = nn.ModuleDict({})
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:

            def lora_dropout_layer(x):
                return x

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.lora_P_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_P_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            #nn.init.zeros_(self.lora_P_B[adapter_name].weight)



class LinearWithLora(nn.Linear, PLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: bool = True,
    ):
        init_lora_weights = init_lora_weights
        nn.Linear.__init__(self, in_features, out_features)
        PLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor, user_embedding = None, mim_list: Optional[list] =  None):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            self.weight = self.weight.to(device=x.device)
            self.bias = nn.Parameter(self.bias.to(device=x.device))
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            if isinstance(user_embedding, torch.Tensor):
                user_embedding = user_embedding.unsqueeze(0)
                #user_embedding = user_embedding.repeat(1,x.shape[1],1)
                user_embedding = user_embedding.repeat(x.shape[0],1,1)
                #result_p = (self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))) + self.lora_P_B[self.active_adapter](self.lora_P_A[self.active_adapter](self.lora_dropout[self.active_adapter](self.lora_dropout[self.active_adapter](user_embedding))))) * self.scaling[self.active_adapter]
                result_p = (self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x)) + self.lora_P_A[self.active_adapter](self.lora_dropout[self.active_adapter](user_embedding))))  * self.scaling[self.active_adapter]
                #result_p = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x)))
                
                # plora
                """
                result_p = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x)) +
                        (0 if p is None else self.lora_P[self.active_adapter](self.lora_dropout[self.active_adapter](p.unsqueeze(1))))
                    )
                    * self.scaling[self.active_adapter]
                )
                """
                # only PKI
                # result_p = (
                #     self.lora_B[self.active_adapter](
                #         (0 if p is None else self.lora_P[self.active_adapter](self.lora_dropout[self.active_adapter](p.unsqueeze(1))))
                #     )
                #     * self.scaling[self.active_adapter]
                # )
                result += result_p
                if mim_list is not None:
                    result_nop = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                    )
                    mim_list.append((result_p, result_nop))
            elif isinstance(user_embedding, dict):
                #result_p = (self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x)) + self.lora_P_A[self.active_adapter](self.lora_dropout[self.active_adapter](self.moe(x, user_embedding).repeat(x.shape[0],1,1).to(self.lora_P_A[self.active_adapter].weight.device)))))  * self.scaling[self.active_adapter]
                result_p = (self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x)) + self.lora_dropout[self.active_adapter](self.moe(x, user_embedding).repeat(x.shape[0],1,1).to(self.lora_B[self.active_adapter].weight.device))))  * self.scaling[self.active_adapter]
                result += result_p
                if mim_list is not None:
                    result_nop = (
                            self.lora_B[self.active_adapter](
                                self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                            )
                            * self.scaling[self.active_adapter]
                    )
                    mim_list.append((result_p, result_nop))
            elif user_embedding is None:
                result_p = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x)))
                result += result_p
            else:
                raise("user_embedding type is not supported")

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Agent(nn.Module):
    def __init__(self,
                 num_relations,
                 model_path, 
                 vae_input_dim, 
                 vae_h_dim, 
                 vae_z_dim,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 n_layers,
                 proj_mid = 10,
                 proj_drop = 0.2,
                 proj_token_num = 1) -> None:
        super().__init__()
        self.sys_prompt = "As a communication expert with outstanding communication habits, you embody the role of CHATBOT throughout the following dialogues. Here are some of your distinctive personal traits: [CHATBOTPERSONA].\n"
        self.chat_prompt = f"""<CONTEXT>\nDrawing from your recent conversation with USER:\n[CONTEXT]\n""" \
                            + f"""<USER_TRAITS>\nDuring the conversation process between you and USER in the past, you found that the USER has the following characteristics:\n[USERPERSONA]\n""" \
                            + f"""\nNow, please role-play as CHATBOT to continue the dialogue between CHATBOT and USER.\n""" \
                            + f"""USER just said: [QUERY]\n""" \
                            + f"""Please respond to USER's statement using the following format (maximum 50 words, must be in English):\nRESPONSE:\n"""

        self.model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map = 'auto')
        self.device = "cuda:0"
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.rgcn = RGCN(in_channels = in_channels, hidden_channels = hidden_channels, out_channels = out_channels, num_relations = num_relations, n_layers = n_layers).to(self.device)
        self.vae = VAE(input_dim = vae_input_dim, h_dim = vae_h_dim, z_dim = vae_z_dim).to(self.device)
        for name, param in self.vae.named_parameters():
            param.requires_grad = False

        self.perosna_proj = nn.Sequential(
            nn.Linear(4096, 4096 * int(proj_mid)),  # self.rec_encoder.config.embedding_size
            nn.ReLU(),
            nn.Dropout(proj_drop),
            nn.Linear(4096 * int(proj_mid), 4096 * proj_token_num), # self.llama_model.config.hidden_size
        ).to(self.device)
        # the code below is used for initializing the representation. We pre-store these vectors in an .npy file to accelerate training.
        self.query_vec_train = np.load("./vector_library/session_4/data_llama/query_driven_train.npy")
        self.node_train = np.load("./vector_library/session_4/triple_from_doubao/node_train_llama.npy")
        self.query_vec_test = np.load("./vector_library/session_4/data_llama/query_driven_test.npy")
        self.node_test = np.load("./vector_library/session_4/triple_from_doubao/node_test_llama.npy")
        with open("./vector_library/session_4/data_llama/query2id_train.json") as f:
            self.query2id_train = json.load(f)
        with open("./vector_library/session_4/triple_from_doubao/node2id_train_llama.json") as f:
            self.node2id_train = json.load(f)
        with open("./vector_library/session_4/data_llama/query2id_test.json") as f:
            self.query2id_test = json.load(f)
        with open("./vector_library/session_4/triple_from_doubao/node2id_test_llama.json") as f:
            self.node2id_test = json.load(f)
        self.model_tokenizer.pad_token = self.model_tokenizer.eos_token
        self.mode = "train"

    def forward(self, data):
        chatbot_embedding_ = self.persona_encoder(data, data['query'])

        query_driven = data['query']
        vae_z,_ = self.vae(query_driven)

        max_input_length = 2048
        max_output_length = 64
        batched_input_ids = []
        batched_labels = []        
        sys_prompt = self.sys_prompt
        chat_prompt = self.chat_prompt
        sys_prompt = sys_prompt.replace("[CHATBOTPERSONA]", "\"" + " ".join(data['init_personas'][1]) + "\"")

        global_context = ""
        for idx, c in enumerate(data['long_memory']):
            c = c.split("CHATBOT")
            global_context = global_context + "(line {}) ".format(2 * idx) + c[0].strip() + "\n"
            global_context = global_context + "(line {}) ".format(2 * idx + 1) + "CHATBOT" + c[1].strip() + "\n"
        
        local_context = ""
        if len(data['short_memory']) > 0:
            for idx, c in enumerate(data['short_memory']):
                c = c.split("CHATBOT")
                local_context = local_context + "(line {}) ".format(2 * idx) + c[0].strip() + "\n"
                local_context = local_context + "(line {}) ".format(2 * idx + 1) + "CHATBOT" + c[1].strip() + "\n"

        if local_context != "":
            context_prompt = "Relevant dialogue from global dialogue history:\n" + global_context + "Dialogue history in current session:\n" + local_context
        else:
            context_prompt = "Relevant dialogue from global dialogue history:\n" + global_context
        
        chat_prompt = chat_prompt.replace("[CONTEXT]",  context_prompt)
        chat_prompt = chat_prompt.replace("[USERPERSONA]", " ".join(data['init_personas'][0]) + "\n")
        chat_prompt = chat_prompt.replace("[QUERY]", data['query'])

        self.model_tokenizer.padding_side = "left"

        prompts_tokens = self.model_tokenizer(
            sys_prompt + chat_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        unk_token_id = self.model_tokenizer.unk_token_id

        to_regress_tokens = self.model_tokenizer(
            data['response'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([prompts_tokens.attention_mask.shape[0], prompts_tokens.attention_mask.shape[1]],dtype=torch.long).to(self.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        input_ids = torch.cat([prompts_tokens.input_ids, to_regress_tokens.input_ids] , dim = 1)
        attention_mask = torch.cat([prompts_tokens.attention_mask, to_regress_tokens.attention_mask], dim = 1)

        user_embedding = vae_z

        inputs_embeds = self.model.model.embed_tokens(input_ids) 
        inputs_embeds = torch.cat([chatbot_embedding_.unsqueeze(0).unsqueeze(0).to(self.model.device), inputs_embeds], dim = 1)
        attention_mask = torch.cat([torch.ones([1, 1],dtype=torch.long).to(self.model.device), attention_mask], dim = 1)
        targets = torch.cat([torch.ones([1, 1],dtype=torch.long).to(self.model.device).fill_(-100), targets], dim = 1)# batch, virtual_nu

        outputs = self.model(
            inputs_embeds = inputs_embeds,
            #input_ids = input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            user_embedding = user_embedding,
            #mim = 1
        )
        # loss
        return {
            "loss": outputs.loss
        }
    
    def generate(self, data):

        chatbot_embedding_ = self.persona_encoder(data, query=data['query'])
        
        query_driven = data['query']
        vae_z, _ = self.vae(query_driven)
        
        batched_input_ids = []
        max_input_length = 2048
        sys_prompt = self.sys_prompt
        chat_prompt = self.chat_prompt
        sys_prompt = sys_prompt.replace("[CHATBOTPERSONA]", "\"" + " ".join(data['chabot_persona']) + "\"")
        sys_prompt = sys_prompt.replace("[CHATBOTPERSONA]", "\"" + " ".join(data['init_personas'][1]) + "\"")

        global_context = ""
        for idx, c in enumerate(data['long_memory']):
            c = c.split("CHATBOT")
            global_context = global_context + "(line {}) ".format(2 * idx) + c[0].strip() + "\n"
            global_context = global_context + "(line {}) ".format(2 * idx + 1) + "CHATBOT" + c[1].strip() + "\n"
        
        local_context = ""
        if len(data['short_memory']) > 0:
            for idx, c in enumerate(data['short_memory']):
                c = c.split("CHATBOT")
                local_context = local_context + "(line {}) ".format(2 * idx) + c[0].strip() + "\n"
                local_context = local_context + "(line {}) ".format(2 * idx + 1) + "CHATBOT" + c[1].strip() + "\n"

        if local_context != "":
            context_prompt = "Relevant dialogue from global dialogue history:\n" + global_context + "Dialogue history in current session:\n" + local_context
        else:
            context_prompt = "Relevant dialogue from global dialogue history:\n" + global_context
        
        chat_prompt = chat_prompt.replace("[CONTEXT]",  context_prompt)
        chat_prompt = chat_prompt.replace("[USERPERSONA]", " ".join(data['init_personas'][0]) + "\n")
        chat_prompt = chat_prompt.replace("[QUERY]", data['query'])

        self.model_tokenizer.padding_side = "left"

        prompts_tokens = self.model_tokenizer(
            sys_prompt + chat_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        user_embedding = vae_z
        inputs_embeds = self.model.model.embed_tokens(prompts_tokens.input_ids) 
        inputs_embeds = torch.cat([chatbot_embedding_.unsqueeze(0).unsqueeze(0).to(self.model.device), inputs_embeds], dim = 1)
        attention_mask = torch.cat([torch.ones([1, 1],dtype=torch.long).to(self.model.device), prompts_tokens.attention_mask], dim = 1)
    
        outputs = self.model.generate(
            #input_ids = prompts_tokens.input_ids,
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens=50,
            user_embedding = user_embedding
        )
        # loss
        return self.model_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def persona_encoder(self, data, query = None):
        chatbot_node_dict = data['node_dict']
        edge_index_chatbot = data['edge_index']
        edge_type_chatbot = data['edge_type']

        chatbot_x = []
        with torch.no_grad():
            if self.mode == 'train':
                query_embedding = torch.from_numpy(self.query_vec_train[self.query2id_train[query]]).unsqueeze(0).to(self.model.device)
            else:
                if query not in self.query2id_test.keys():
                    query = self.model_tokenizer(query, return_tensors='pt').to(self.model.device)
                    query_outputs = self.model.model(**query, output_hidden_states = True)
                    query_embeddings = list(query_outputs.hidden_states)
                    query_last_hidden_states = query_embeddings[-1].cpu().numpy()
                    query_first_hidden_states = query_embeddings[1].cpu().numpy()
                    query_last_hidden_states = np.squeeze(query_last_hidden_states)
                    query_first_hidden_states = np.squeeze(query_first_hidden_states)
                    query_embedding = torch.from_numpy(np.mean(query_first_hidden_states + query_last_hidden_states, axis=0)).unsqueeze(0).to(self.model.device)
                    
                else:
                    query_embedding = torch.from_numpy(self.query_vec_test[self.query2id_test[query]]).unsqueeze(0).to(self.model.device)

            for node in chatbot_node_dict:
                if self.mode == 'train':
                    embedding = torch.from_numpy(self.node_train[self.node2id_train[node]]).unsqueeze(0).to(self.model.device)
                else:
                    if node not in self.node2id_test.keys():
                        node = self.model_tokenizer(node, return_tensors='pt').to(self.model.device)
                        outputs = self.model.model(**node, output_hidden_states = True)
                        embeddings = list(outputs.hidden_states)
                        last_hidden_states = embeddings[-1].cpu().numpy()
                        first_hidden_states = embeddings[1].cpu().numpy()
                        last_hidden_states = np.squeeze(last_hidden_states)
                        first_hidden_states = np.squeeze(first_hidden_states)
                        embedding = torch.from_numpy(np.mean(first_hidden_states + last_hidden_states, axis=0)).unsqueeze(0).to(self.model.device)
                    else:
                        embedding = torch.from_numpy(self.node_test[self.node2id_test[node]]).unsqueeze(0).to(self.model.device)
                #embedding = torch.mean(self.model.transformer(**node)[0].squeeze(1), dim = 0).unsqueeze(0)
                chatbot_x.append(embedding)

        chatbot_x = torch.stack(chatbot_x).squeeze(1).to(self.device)
        edge_index_chatbot = torch.LongTensor(edge_index_chatbot).to(self.device)
        edge_type_chatbot = torch.LongTensor(edge_type_chatbot).squeeze(1).to(self.device)
        rgcn_output_chatbot = self.rgcn(chatbot_x, edge_index_chatbot, edge_type_chatbot, query = query_embedding)
        #chatbot_embedding = self.perosna_proj(rgcn_output_chatbot)
        #return rgcn_output_chatbot[0,:], chatbot_embedding[0,:]
        return rgcn_output_chatbot[0,:]


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--log_name", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--epoches",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--vae_input_dim",type=int)
    parser.add_argument("--vae_h_dim",type=int)
    parser.add_argument("--vae_z_dim",type=int)
    parser.add_argument("--in_channels",type=int)
    parser.add_argument("--hidden_channels",type=int)
    parser.add_argument("--out_channels",type=int)
    parser.add_argument("--n_layers",type=int)
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_name,
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.DEBUG)

    with open(args.data_path,'r',encoding='utf-8') as f:
        dataset = json.load(f)
    
    with open(args.data_path,'r',encoding='utf-8') as f:
        dataset = json.load(f)
    # You can change the different sessions in the code below.
    with open("./entity_link/session_4/train_triple_context.json",'r',encoding='utf-8') as f: # Persona triples from the historical context that appear in the training set of session 4.
        context_triple = json.load(f)

    with open("./entity_link/session_4/train_triple_persona.json",'r',encoding='utf-8') as f: # Persona triples from the defined personas that appear in the training set of session 4.
        persona_triple = json.load(f)

    with open("./entity_link/session_4/train_triple_query.json",'r',encoding='utf-8') as f: # Persona triples from current query that appear in the training set of session 4.
        query_triple = json.load(f)
    # The persona triples of the current query and defined personas in the training set of session 4, are expanded triples based on the knowledge graph.
    with open("./entity_link/session_4/peacok_linked_query_and_persona_train.json",'r',encoding='utf-8') as f: 
        query_persona_kg_extend = json.load(f)
    # The persona triples of context in the training set of session 4, are expanded triples based on the knowledge graph.
    with open("./entity_link/session_4/peacok_linked_context_train.json",'r',encoding='utf-8') as f:
        context_kg_extend = json.load(f)
    
    model = Agent(len(relation2id), args.model_path, args.vae_input_dim, args.vae_h_dim, args.vae_z_dim, args.in_channels, args.hidden_channels, args.out_channels, args.n_layers)
    target_modules = ['q_proj', 'k_proj', 'v_proj']
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in target_modules)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                new_module = LinearWithLora(adapter_name = 'default', in_features = in_features, out_features = out_features, r = 16, lora_alpha=32, lora_dropout=0.1).to(target.weight.device)
                for name, p in new_module.named_parameters():
                    p.requires_grad = False
                _replace_module(parent, target_name, new_module, target)

    
    model = load_checkpoint(model = model, checkpoint_path = args.checkpoint_path)# load vae-lora checkpoint
    optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-5,
                weight_decay=1e-3,
                betas=(0.9, 0.99), 
            )
    
    epoches = args.epoches
    batch_size = args.batch_size
    loss_all = 0
    for epoch in range(epoches):
        cnt = 0
        logging.info("Start training epoch {}".format(epoch))
        model.train()
        model.mode = 'train'
        for data in tqdm.tqdm(dataset):
            data = get_train_data(data, context_triple, persona_triple, query_triple, query_persona_kg_extend, context_kg_extend, session_num = 4)
            cnt = cnt + 1
            try:
                loss = model(data)["loss"]
                loss = loss / 32
                loss.backward()
                if cnt % 32 == 0:
                    logging.info("epoch:{}, iters:{}, learning_rate:{}, loss:{}".format(epoch, cnt, optimizer.state_dict()['param_groups'][0]['lr'], loss_all))
                    loss_all = 0
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    loss_all = loss_all + loss.item()
            except:
                cnt = cnt - 1
            
            with torch.cuda.device('cuda:0'):
                torch.cuda.empty_cache()
        save_checkpoint(model = model, optimizer = optimizer, cur_epoch = epoch)

if __name__ == '__main__':
    main()