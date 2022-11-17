from torch import nn
from torch.nn import Parameter
from torch.nn import init
from utils import *
import dgl
from dgl.nn.pytorch.hetero import HeteroGraphConv
from dgl.nn.pytorch.conv.graphconv import GraphConv
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 kernel_initializer=None,
                 bias_initializer='Zero'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        if kernel_initializer is not None:
            self.reset_parameters(self.linear.weight, kernel_initializer)
        if bias and bias_initializer is not None:
            self.reset_parameters(self.linear.bias, bias_initializer)

    @staticmethod
    def reset_parameters(param, initializer):
        if initializer == 'Xavier_Uniform':
            init.xavier_uniform_(param, gain=1.)
        elif initializer == 'Xavier_Normal':
            init.xavier_normal_(param, gain=1.)
        elif initializer == 'Kaiming_Uniform':
            init.kaiming_uniform_(param)
        elif initializer == 'Kaiming_Normal':
            init.kaiming_normal_(param, a=1.)
        elif initializer == 'Uniform':
            init.uniform_(param, a=0, b=1)
        elif initializer == 'Normal':
            init.normal_(param, mean=0, std=1)
        elif initializer == 'Orthogonal':
            init.orthogonal_(param, gain=1)
        elif initializer == 'Zero':
            init.zeros_(param)
        elif initializer == 'gcn':
            stdv = 1. / math.sqrt(param.size(1))
            param.data.uniform_(-stdv, stdv)
        # elif initializer == 'Uniform4Bias':
        #     fan_in, _ = self.weight.shape
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(param, -bound, bound)

    def forward(self, x):
        return self.linear(x)


class MotifConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 compress_dim,
                 rel_names,
                 dataset,
                 motif_mats,
                 mw_init,
                 att_act,
                 motif_dropout,
                 att_dropout,
                 aggr='mean',
                 root_weight=True,
                 bias=True):
        """

        Args:
            in_channels: (int): Size of each input sample.
            out_channels:
            rel_names:
            root_weight:
        """
        super(MotifConv, self).__init__()
        self.rel_names = rel_names
        self.aggr = aggr
        self.motif_mats = motif_mats
        self.dataset = dataset
        self.mw_init = mw_init
        self.att_act = att_act
        self.motif_dropout = motif_dropout
        self.att_dropout = att_dropout
        d = compress_dim

        self.conv = HeteroGraphConv({
            rel: GraphConv(in_channels, out_channels, norm='none', weight=False, bias=False)
            for rel in rel_names
        }, aggregate='sum')

        self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_channels, out_channels))

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

        else:
            raise Exception('Unknown motif param initial method')

        zeros(self.bias)
        zeros(self.ba)

        for b in self.motif_biases:
            zeros(b)

    def forward(self, g, inputs):
        g = g.local_var()

        wdict = {self.rel_names[i]: {'weight': w.squeeze(0), 'edge_weight': g.edges[self.rel_names[i]].data['edge_weight_norm']}
                 for i, w in enumerate(torch.split(self.weight, 1, dim=0))}

        hs = self.conv(g, {'P': inputs}, mod_kwargs=wdict)

        h = hs['P']
        if self.aggr == 'mean':
            degs = dgl.to_homogeneous(g).in_degrees()
            norm = 1.0 / degs
            norm = torch.where(torch.isinf(norm), torch.full_like(norm, 0.), norm).float()
            shp = norm.shape + (1,) * (h.dim() - 1)
            norm = torch.reshape(norm, shp)
            h = h * norm

        if self.root is not None:
            h += torch.matmul(inputs, self.root)

        if self.bias is not None:
            h += self.bias

        motif_rsts = [h]
        for i in range(13):
            motif_rsts.append(torch.spmm(self.motif_mats[i], h))

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


