import json
import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, BertForSequenceClassification
from memory import BiEncoderRetriever
from tkinter import _flatten
import argparse
import os

class Topic(nn.Module):
    def __init__(self, model_path, checkpoint_path) -> None:
        super().__init__()
        self.model = BertModel.from_pretrained(model_path)
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self.load_checkpoint(checkpoint_path)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = 2
        self.classifier = nn.Linear(3 * 768, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, data):
        input_a = self.model_tokenizer(data['a_sentence'], return_tensors='pt').to(self.model.device)
        outputs_a = self.model(**input_a)
        labels = torch.LongTensor([data['label']]).to(self.model.device)
        pooled_output_a = outputs_a[1]
        pooled_output_a = self.dropout(pooled_output_a)

        input_b = self.model_tokenizer(data['b_sentence'], return_tensors='pt').to(self.model.device)
        outputs_b = self.model(**input_b)
        pooled_output_b = outputs_b[1]
        pooled_output_b = self.dropout(pooled_output_b)

        for_classifier = torch.cat([pooled_output_a, pooled_output_b, torch.abs(pooled_output_a - pooled_output_b)], dim = 1)
        logits = torch.softmax(self.classifier(for_classifier), dim=-1)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "logits": logits,
            "loss": loss
        }

    def load_checkpoint(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--output_dir",type=str)
    args = parser.parse_args()
    model = BiEncoderRetriever()
    topic_detector = Topic(args.model_path, args.checkpoint_path)
    for s in range(2,5):
        for mode in ['train','test']:
            path = args.data_path + "/session_" + str(s) + "/" + mode + ".json" # concat your data path
            with open(path,'r',encoding='utf-8') as f:
                dataset = json.load(f)
            res = []
            for data in tqdm.tqdm(dataset):
                context = data['context']
                long_term = context[:s-1]
                short_term = context[s-1]
                # Retrieval and re-rank long-term memory
                memory_bank = []
                for session in long_term:
                    for utterance_num in range(0, len(session), 2):
                        if utterance_num + 1 >= len(session):
                            continue
                        momery = session[utterance_num] + "[SEP]" + session[utterance_num + 1] 
                        memory_bank.append(momery)
                Long_memory = model.dial_sort(model.retrieve_top_summaries(question = data['query'], summaries = list(_flatten(memory_bank))))
                tmp_data = data
                tmp_data['long_memory'] = ["USER: " + context.split("[SEP]")[0] + " CHATBOT: " + context.split("[SEP]")[1]  for context in Long_memory]
                short_memory = []
                # topic detectot to obtain shory_term memory
                for utterance_num in range(0, len(short_term), 2):
                    if utterance_num + 1 >= len(session):
                        continue
                    momery = short_term[utterance_num] + "[SEP]" + short_term[utterance_num + 1] 
                    logits = model({"a_sentence": data['query'], "b_sentence": momery, "label": data['label']})['logits']
                    predicted = torch.max(logits, dim = 1).indices
                    if predicted == 1:
                        short_memory.append(memory_bank)
                    else:
                        short_memory = []
                tmp_data['short_memory'] = ["USER: " + context.split("[SEP]")[0] + " CHATBOT: " + context.split("[SEP]")[1]  for context in short_memory]
                res.append(tmp_data)

            output_path = args.output_dir + "/session_" + str(s) + "/" + mode + "_processed.json"
            with open("./session_5/test_data_with_context_and_memory.json",'w',encoding='utf-8') as f:
                f.write(json.dumps(res, indent=4, ensure_ascii=False))
