from torch import nn
import torch
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer

class VAE(nn.Module):

    def __init__(self, input_dim=4096, h_dim=4096, z_dim=4096):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.encoder = BertModel.from_pretrained("/home/data/public/bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/data/public/bert-base-uncased")
        self.fc1 = nn.Linear(768, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var
        self.mim_loss = nn.MSELoss()

    def forward(self, query_enc, response_enc = None):

        # encoder
        mu_q, log_var_q = self.encode(query_enc)
        # reparameterization trick
        sampled_z_q = self.reparameterization(mu_q, log_var_q)
        kld_loss = -0.5 * torch.mean(
            (1 + log_var_q - mu_q**2 -
             torch.exp(log_var_q)),
            dim=1)
        # sum the instance loss
        kld_loss = torch.mean(kld_loss, dim=0)

        if response_enc is not None:
            mu_r, log_var_r = self.encode(response_enc)
            sampled_z_r = self.reparameterization(mu_r, log_var_r)
            mim_loss = self.mim_loss(sampled_z_q, sampled_z_r)
            loss = kld_loss + mim_loss
        else:
            loss = kld_loss
        return sampled_z_q, loss

    def encode(self, x):
        x_tokens = self.tokenizer(x, return_tensors='pt').to(self.encoder.device)
        encoder_vec = self.encoder(**x_tokens)['pooler_output']
        h = F.relu(self.fc1(encoder_vec))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        """sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  """
        sigma = torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(sigma)
        z_s = mu + sigma * eps
        return z_s

