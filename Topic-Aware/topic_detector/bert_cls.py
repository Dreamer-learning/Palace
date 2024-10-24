import json
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, BertForSequenceClassification
import tqdm
import logging
import os
import argparse
logging.basicConfig(filename='topic_shift_bert.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.DEBUG)


def save_checkpoint(model, optimizer, cur_epoch, output_dir):
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
        output_dir,
        "checkpoint_{}.pth".format(cur_epoch),
    )
    logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)

def load_checkpoint(model):
    checkpoint_path = os.path.join("/home/liudongshuo/topic_shift/checkpoint", "checkpoint_0.pth")
    logging.info("Loading checkpoint from {}.".format(checkpoint_path))
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


class Topic(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.model = BertModel.from_pretrained(model_path)
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_path)

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


def evaluation(model, test_dir):
    model.eval()
    with open(test_dir,'r',encoding='utf-8') as f:
        dataset = json.load(f)

    cnt = 0
    for data in tqdm.tqdm(dataset):
        a_sentence, b_sentence = data['input'].split('[SEP]')
        logits = model({"a_sentence": a_sentence, "b_sentence": b_sentence, "label": data['label']})['logits']
        predicted = torch.max(logits, dim = 1).indices
        if predicted == data['label']:
            cnt = cnt + 1
    logging.info("acc:{}".format(cnt / len(dataset)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="train dataset path")
    parser.add_argument("--test_dir", type=str, help="test dataset path")
    parser.add_argument("--epoches", type=int)
    parser.add_argument("--lr",type=float, help="learning rate")
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--output_dir",type=str)
    args = parser.parse_args()
    model = Topic(args.model_path)
    with open(args.train_dir,'r',encoding='utf-8') as f:
        dataset = json.load(f)
    epoches = args.epoches
    optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-4,
                betas=(0.9, 0.99), 
            )
    for epoch in range(epoches):
        cnt = 0
        loss_all = 0
        logging.info("Start training epoch {}".format(epoch))
        model.train()
        for data in tqdm.tqdm(dataset):
            cnt = cnt + 1 
            a_sentence, b_sentence = data['input'].split('[SEP]')
            loss = model({"a_sentence": a_sentence, "b_sentence": b_sentence, "label": data['label']})['loss']
            loss = loss / 32
            loss.backward()
            if cnt % 32 == 0:
                logging.info("epoch:{}, iters:{}, learning_rate:{}, loss:{}".format(epoch, cnt, optimizer.state_dict()['param_groups'][0]['lr'], loss_all))
                loss_all = 0
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss_all = loss_all + loss.item()
        save_checkpoint(model = model, optimizer = optimizer, cur_epoch = epoch,output_dir=args.output_dir)
        evaluation(model = model, test_dir=args.test_dir)

if __name__ == '__main__':
    main()