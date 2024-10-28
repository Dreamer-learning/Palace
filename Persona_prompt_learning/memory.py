import torch
import numpy as np

from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder
from typing import List

class BiEncoderRetriever:
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("/home/data/public/dpr-multidoc2dial-structure-question-encoder")
        self.question_encoder = DPRQuestionEncoder.from_pretrained("/home/data/public/dpr-multidoc2dial-structure-question-encoder").to(self.device)
        self.ctxt_encoder = DPRContextEncoder.from_pretrained("/home/data/public/dpr-multidoc2dial-structure-ctx-encoder").to(self.device)
        for name, param in self.question_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.ctxt_encoder.named_parameters():
            param.requires_grad = False

    def encode_summaries(self, summaries: List[str]):
        input_dict = self.tokenizer(summaries, padding='max_length', max_length=64, truncation=True, return_tensors="pt").to(self.device)
        del input_dict["token_type_ids"]
        return self.ctxt_encoder(**input_dict)['pooler_output']

    def encode_question(self, question: str):
        input_dict = self.tokenizer(question, padding='max_length', max_length=32, truncation=True, return_tensors="pt").to(self.device)
        del input_dict["token_type_ids"]
        return self.question_encoder(**input_dict)['pooler_output']

    def retrieve_top_summaries(self, question: str, summaries: List[str], encoded_summaries: np.ndarray = None, topk: int = 5):
        encoded_question = self.encode_question(question)
        if encoded_summaries is None:
            encoded_summaries = self.encode_summaries(summaries)
        else:
            encoded_summaries = torch.from_numpy(encoded_summaries).to(self.device)

        scores = torch.mm(encoded_question, encoded_summaries.T)
        if topk >= len(summaries):
            return summaries
        top_k = torch.topk(scores, topk).indices.squeeze()
        return [summaries[i] for i in top_k if scores[0,i] > 80]  