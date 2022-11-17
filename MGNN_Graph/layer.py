from torch import nn
import torch
from torch.nn import Parameter
from torch.nn import init
from utils import *
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_sparse import matmul
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class GCConv(MessagePassing):
    def __init__(self, **kwargs):
        super(GCConv, self).__init__(**kwargs)

    def forward(self, edge_index, x, edge_weight):
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MotifConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 compress_dim,
                 mw_init,
                 att_act,
                 motif_dropout,
                 att_dropout,
                 device,
                 root_weight=True,
                 bias=True):
        """

        Args:
            in_channels: (int): Size of each input sample.
            out_channels:
            root_weight:
        """
        super(MotifConv, self).__init__()
        self.mw_init = mw_init
        self.att_act = att_act
        self.motif_dropout = motif_dropout
        self.att_dropout = att_dropout
        self.device = device
        d = compress_dim

        self.conv = GCConv()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.wa = nn.Parameter(torch.Tensor(out_channels, d))
        self.ba = nn.Parameter(torch.Tensor(d))

        self.wb = nn.Parameter(torch.Tensor(out_channels, d))

        self.motif_weights = nn.ParameterList(
            [nn.Parameter(torch.Tensor(out_channels * 13, d)) for _ in range(13)]
        )

        self.motif_biases = nn.ParameterList(
            [nn.Parameter(torch.Tensor(d)) for _ in range(13)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        init.kaiming_uniform_(self.root)

        if self.mw_init == 'Kaiming_Uniform':
            init.kaiming_uniform_(self.wa)
            init.kaiming_uniform_(self.wb)

            for w in self.motif_weights:
                init.kaiming_uniform_(w)

        elif self.mw_init == 'Xavier_Normal':
            init.xavier_normal_(self.wa)
            init.xavier_normal_(self.wb)

            for w in self.motif_weights:
                init.xavier_normal_(w)

        elif self.mw_init == 'Xavier_Uniform':
            init.xavier_uniform_(self.wa)
            init.xavier_uniform_(self.wb)

            for w in self.motif_weights:
                init.xavier_uniform_(w)

        elif self.mw_init == 'Orthogonal':
            init.orthogonal_(self.wa)
            init.orthogonal_(self.wb)

            for w in self.motif_weights:
                init.orthogonal_(w)

        else:
            raise Exception('Unknown motif param initial method')

        zeros(self.bias)
        zeros(self.ba)

        for b in self.motif_biases:
            zeros(b)

    def forward(self, graphs, inputs):
        h = inputs
        # h = torch.mm(inputs, self.weight)

        graph0 = graphs[0].to(self.device)

        h = self.conv(graph0.edge_index, h, edge_weight=graph0.edge_weight4norm)
        h = torch.mm(h, self.weight)

        if self.root is not None:
            h += torch.matmul(inputs, self.root)

        if self.bias is not None:
            h += self.bias

        motif_rsts = [h]
        for i in range(1, 14):
            motif = graphs[i].to(self.device)

            motif_rsts.append(self.conv(motif.edge_index, h, motif.edge_weight4motif))

        motif_embeds = []
        for i in range(1, 14):
            compress_list = motif_rsts[:i] + motif_rsts[i+1:]
            mw = torch.mm(motif_rsts[i], F.dropout(self.wa, p=self.motif_dropout, training=self.training)) + self.ba
            c = torch.mm(torch.cat(compress_list, dim=1), self.motif_weights[i-1]) + self.motif_biases[i-1]
            att = self.att_act(torch.sum(mw * c, dim=1, keepdim=True))
            att = F.dropout(att, p=self.att_dropout, training=self.training)
            motif_embeds.append(att * (mw - c))

        outputs = torch.cat(motif_embeds, dim=1)

        return outputs


if __name__ == '__main__':
    pass


