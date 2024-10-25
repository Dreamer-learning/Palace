import torch
import numpy as np

from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder
from typing import List
import torch.nn.functional as F

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

    def retrieve_top_summaries(self, question: str, summaries: List[str], encoded_summaries: np.ndarray = None, topk: int = 3):
        encoded_question = self.encode_question(question)
        if encoded_summaries is None:
            encoded_summaries = self.encode_summaries(summaries)
        else:
            encoded_summaries = torch.from_numpy(encoded_summaries).to(self.device)

        scores = torch.mm(encoded_question, encoded_summaries.T)
        if topk >= len(summaries):
            return summaries
        top_k = torch.topk(scores, topk).indices.squeeze()
        return [summaries[i] for i in top_k]

    def dial_rel_score(self, dial1, dial2, alpha=1):
        dial1_vec = []
        dial2_vec = []
        input_diag1 = self.tokenizer(dial1, padding='max_length', max_length=64, truncation=True, return_tensors="pt").to(self.device)
        vec = self.ctxt_encoder(**input_diag1)['pooler_output']
        dial1_vec.append(('Diag1', vec))
        
        input_diag2 = self.tokenizer(dial2, padding='max_length', max_length=64, truncation=True, return_tensors="pt").to(self.device)
        vec = self.ctxt_encoder(**input_diag2)['pooler_output']
        dial2_vec.append(('Diag2', vec))
        turn1, turn2 = len(dial1_vec), len(dial2_vec)
        dp = [[0] * (turn1+1)] * (turn2+1)
        for i in range(turn1):
            for j in range(turn2):
                if i == 0:
                    dp[i][j] = j  
                elif j == 0:
                    dp[i][j] = i
                else:
                    dp[i][j] = min(1+dp[i][j-1], # insertion
                                1+dp[i-1][j], # deletion
                                alpha * (1-F.cosine_similarity(dial1_vec[i-1][1], dial2_vec[j-1][1])) if dial1_vec[i-1][0] == dial1_vec[j-1][0] else 1e9 + 7 + dp[i-1][j-1]) # substitution
        return dp[turn1][turn2]
    # dynamic
    def dial_sort(self, dials):
        length = len(dials)
        dis = []
        for i in range(length):
            dis.append([])
            for j in range(length):
                if i == j:
                    dis[i].append(0)
                else:
                    rel = self.dial_rel_score(dials[i], dials[j])
                    dis[i].append(rel)
        visited = [False] * length
        path = []
        cur_index = 0
        visited[cur_index] = True
        path.append(dials[cur_index])

        while len(path) < length:
            min_distance = 100
            next_index = None
            for i in range(length):
                if not visited[i] and dis[cur_index][i] < min_distance:
                    min_distance = dis[cur_index][i]
                    next_index = i
            visited[next_index] = True
            path.append(dials[next_index])
            cur_index = next_index

        return path[1:]

