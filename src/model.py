
import torch.nn as nn
import torch


# Embedding layer
class Embedding(nn.Module):

    def __init__(self, nfeat, nemb):
        """
        :param nfeat: number of features,
            for sql, it == all combinations of filter condiation
            for dataset, it == pre-defined features.
        :param nemb: hyperparameter.
        """
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward_sql(self, x):
        """
         :param x:   {x: LongTensor B*1}
         :return:    embeddings B*F*E
         """
        emb = self.embedding(x)  # B*1*E
        return emb  # B*1*E

    def forward_moe(self, x):
        """
        B: batch size, F: fileds, E: nemb
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    embeddings B*F*E
        """
        # convert feature to embedding.
        emb = self.embedding(x['id'])                           # B*F*E
        # scale by value
        output = emb * x['value'].unsqueeze(2)                  # B*F*E
        return output


# MLP layer
class MLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, dropout, noutput):
        """
        :param ninput: input dimension
        :param nlayers: # hidden layer
        :param nhid: # hidden layer output
        :param dropout: dropout rate
        :param noutput: output dimension
        """
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayers == 0:
            nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)


# MOELayer
class MOELayer(nn.Module):

    duplayers = 0

    def __init__(self, ninput, nhid, dropout):
        """
        :param ninput: input dimension
        :param nhid: # hidden layer output
        :param dropout: dropout rate
        """
        if MOELayer.duplayers == 0:
            raise "Please set hyper K (number of duplication) "

        # dynamically update this.
        self.arch_weight = None     # B*1*K where K is the replication factor.

        super().__init__()
        self.layer_group = list()
        for i in range(MOELayer.duplayers):
            layers = list()
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            self.layer_group.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Reconstruct layer weight and Forward
        :param x: embedding output, B*ninput
        :return: FloatTensor B*nouput
        """
        # 1. generate the output for each layer
        output = list()
        for layer in self.layer_group:
            each_layer_out = layer(x)         # B * nhid
            output.append(each_layer_out)     # K * (B * nhid)
        # stack the list along a new dimension to create a tensor of shape B*K*nhid
        group_layer_output = torch.stack(output, dim=1)

        # 2. weighted sum
        weighted_output = torch.bmm(self.arch_weight, group_layer_output)     # B * nhid
        return weighted_output


# MOEMLP
class MOEMLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, dropout, noutput=1):
        """
        :param ninput: input dimension
        :param nlayers: # hidden MOELayer
        :param nhid: output dimension of hidden MOELayer
        :param dropout: dropout rate
        :param noutput: output dimension of the MOENet
        """
        super().__init__()
        # build MOELayer
        self.layers = list()
        for i in range(nlayers):
            self.layers.append(MOELayer(ninput, nhid, dropout))
            ninput = nhid

        # last layer is not MOELayer
        self.layers.append(nn.Linear(nhid, noutput))
        self.moe_net = nn.Sequential(*self.layers)

    def forward(self, x, arch_weights):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """

        assert arch_weights.shape[1] == len(self.layers)

        # update each moe layer's weight
        for index in range(arch_weights.shape[1]):  # Iterating over L, moe_num_layers
            self.layers[index].arch_weight = x[:, index, :].unsqueeze(1)  # sub_tensor has shape (B, 1, K)

        # then compute
        return self.moe_net(x)


class HyperNet(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self,
                 nfield: int, nfeat: int,
                 nemb: int, nlayers: int, hid_layer_len: int,
                 dropout: float,
                 L: int, K: int):
        """
        :param nfield: for SQL combination, each is one feature, so it's 1
        :param nfeat: number of filter combinations. 'select...where...'
        :param nemb: hyperparams, 10
        :param nlayers: hyperparams 3
        :param hid_layer_len: hyperparams, hidden layer length
        :param dropout: hyperparams 0
        :param L: number of layers in MOENet
        :param K: number of replication for each MOELayer
        """
        super().__init__()
        noutput = L*K
        self.embedding = Embedding(nfeat, nemb)
        self.mlp_ninput = nfield*nemb
        self.mlp = MLP(self.mlp_ninput, nlayers, hid_layer_len, dropout, noutput)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.forward_sql(x)                           # B*F*E
        y = self.mlp(x_emb.view(-1, self.mlp_ninput))       # B*1
        return y.squeeze(1)


class MOENet(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self,
                 nfield: int, nfeat: int,
                 nemb: int, moe_num_layers: int, moe_hid_layer_len: int,
                 dropout: float):
        """
        :param nfield: # used column of the dataset
        :param nfeat: # feature of the dataset.
        :param nemb: hyperparams, embedding size 10
        :param moe_num_layers: hyperparams: # hidden MOElayers of MOENet
        :param moe_hid_layer_len: hyperparams: hidden layer length in MoeLayer
        :param dropout: hyperparams 0
        """
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.moe_mlp_ninput = nfield * nemb
        self.moe_mlp = MOEMLP(self.moe_mlp_ninput, moe_num_layers, moe_hid_layer_len, dropout)

    def forward(self, x, arch_weights: torch.Tensor):
        """

        :param x: {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :param arch_weights: B*L*K
        :return: y of size B, Regression and Classification (+sigmoid)
        """

        x_emb = self.embedding.forward_moe(x)    # B*F*E
        y = self.moe_mlp.forward(
            x=x_emb.view(-1, self.mlp_ninput),   # B*1
            arch_weights=arch_weights,            # B*1*K
        )
        return y.squeeze(1)

