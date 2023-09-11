from copy import deepcopy
import torch
import torch.nn as nn


import argparse
from src.model.gate_network import SparseVerticalGate

from src.model.verticalMoE import VerticalDNN
from third_party.utils.model_utils import EntmaxBisect






class SparseMax_VerticalSAMS(nn.Module):
    
    def __init__(self, args:argparse.Namespace):
        super().__init__()
        
        self.args = args
        
        col_cardinality_sum = args.nfield + args.nfeat + 1
        self.gate_input_size = args.nfield * args.sql_nemb
        
        self.input_size = args.nfield * args.data_nemb
        self.hidden_size = args.hid_layer_len
        self.output_size = args.output_size
        self.num_experts = args.K
        self.alpha = args.alpha
        
        self.gate_network = SparseVerticalGate(self.gate_input_size, self.num_experts ,args.hid_layer_len, args.dropout)
        expert = VerticalDNN(self.input_size, self.output_size, self.hidden_size, args.dropout)
        self.experts = nn.ModuleList([deepcopy(expert) for _ in range(self.num_experts)])

        
        self.sql_embedding = nn.Embedding(col_cardinality_sum, args.sql_nemb)
        self.input_embedding = nn.Embedding(args.nfeat, args.data_nemb)

        nn.init.xavier_uniform_(self.sql_embedding.weight)
        nn.init.xavier_uniform_(self.input_embedding.weight)

        self.sparsemax = EntmaxBisect(alpha=self.alpha)
        
        
        
        ## noisy gating
        # self.w_noise = nn.Parameter(torch.zeros(self.gate_input_size, self.num_experts), requires_grad=True)
        # self.softplus = nn.Softplus()
        # self.noisy_gating = args.noise_gating
        # self.training = False
        

    def cal_gate_score(self, sql:torch.Tensor):
        B = sql.shape[0]
        sql_emb = self.sql_embedding(sql).view(B, -1)
        gate_score = self.gate_network(sql_emb)
        gate_score = self.sparsemax(gate_score)
        return gate_score
    
    
    def forward(self, x:torch.Tensor, sql:torch.Tensor):
        """
        Args:
        x:      [B, nfield] only Id
        sql:    [B, nfield]
        Returns:
        y:      [B]
        """
        x_emb = self.input_embedding(x)         
        sql_emb = self.sql_embedding(sql)
        
        B = x.shape[0]
        x_emb = x_emb.view(B, -1)
        sql_emb = sql_emb.view(B, -1)
        
        gate_score = self.gate_network(sql_emb)
        gate_score = self.sparsemax(gate_score)
        
        # gate_score, _ = self.cal_noise_gate(sql_emb, gate_score, self.training)
        # [B, N]
        
        # --------- Calculate Importance -----------
        importance = gate_score.sum(0)
        
        loss = self.cv_squared(importance)
        
        x_list = [ expert(x_emb) for expert in self.experts]
        x = torch.stack(x_list, dim = 1) 
        # [B, N, nhid]
        
        # sum according to weight
        x = gate_score.unsqueeze(-1) * x
        # [B, N, nhid]
        
        x = torch.sum(x, dim = 1)
        # [B, nhid]
        
        return x.squeeze(-1), loss
    
    
    def train(self, mode=True):
        """
        Overwrite the train function to customize behavior during training.
        """
        # Your custom logic here
        super(SparseMax_VerticalSAMS, self).train(mode)
        self.training = mode

    def eval(self):
        """
        Overwrite the eval function to customize behavior during evaluation.
        """
        # print(f"set eval()")
        # Your custom logic here
        super(SparseMax_VerticalSAMS, self).eval()
        self.training = False 
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    # def cal_noise_gate(self, x, gates, train, noise_epsilon = 1e-2):
        
    #     clean_logits = gates
    #     if self.noisy_gating and train:
    #         raw_noise_stddev = x @ self.w_noise
    #         noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
    #         noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
    #         logits = noisy_logits
    #     else:
    #         logits = clean_logits
        
    #     gate = self.sparsemax(logits)
    #     load = 0
    #     # if self.noisy_gating and train:
    #     return gate, load
            