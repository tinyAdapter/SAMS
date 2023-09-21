import argparse
import torch
import torch.nn as nn
from src.model.expert import VerticalDNN
from src.model.gate_network import SparseVerticalGate, VerticalGate
from src.model.sparseMoE import SparseMoE



    



class VerticalSAMS(nn.Module):
    
    def __init__(self, args:argparse.Namespace):
        super().__init__()
        self.args = args
        
        col_cardinality_sum = args.nfield + args.nfeat + 1
        self.gate_input_size = args.nfield * args.sql_nemb
        # self.gate_network = VerticalGate(self.gate_input_size, args.K, args.hid_layer_len)
        self.gate_network = SparseVerticalGate(self.gate_input_size, args.K, args.hid_layer_len, args.dropout)
        self.input_size = args.nfield * args.data_nemb
        self.hidden_size = args.hid_layer_len
        self.output_size = args.output_size
        self.num_experts = args.K
        self.select_experts = args.C
        
        assert args.C <= args.K
        expert = VerticalDNN(self.input_size, self.output_size, self.hidden_size, dropout=args.dropout)
        self.backbone = SparseMoE( 
                                expert = expert, 
                                # input_size = self.input_size,
                                # output_size = self.output_size,
                                # hidden_size= self.hidden_size,
                                num_experts = self.num_experts,
                                select_experts = self.select_experts,
                                noisy_gating = args.noise_gating,
                                )
        
        self.sql_embedding = nn.Embedding(col_cardinality_sum, args.sql_nemb)
        self.input_embedding = nn.Embedding(args.nfeat, args.data_nemb)

        nn.init.xavier_uniform_(self.sql_embedding.weight)
        nn.init.xavier_uniform_(self.input_embedding.weight)

    
    def cal_gate_score(self, sql:torch.Tensor):
        B = sql.shape[0]
        sql_emb = self.sql_embedding(sql).view(B, -1)
        gate_score = self.gate_network(sql_emb)
        
        top_indices = torch.topk(gate_score, k=self.select_experts, dim = 1).indices
        result = torch.zeros_like(gate_score)
        result.scatter_(1, top_indices, gate_score.gather(1, top_indices))
        return result
    
    def forward(self, x:torch.Tensor, sql:torch.Tensor):
        """
        Args:
        x:      [B, nfield] only Id
        sql:    [B, nfield]
        Returns:
        y:      [B]
        loss:   [1]
        """
        
        x_emb = self.input_embedding(x)             # [B, nfield, data_nemb]
        sql_emb = self.sql_embedding(sql)           # [B, nfield, sql_nemb]
        B = x.shape[0]
        
        x_emb = x_emb.view(B, -1)                   # [B, input_size]
        sql_emb = sql_emb.view(B, -1)               # [B, gate_input_size]

        gate_scores = self.gate_network(sql_emb)    # [B, num_experts]
        y, loss = self.backbone(x_emb, gate_scores) 
        return y.squeeze(-1), loss 