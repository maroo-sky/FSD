import pdb
import torch
from torch import nn
from metrics.LayerWiseMetrics import cdist2
import torch.nn.functional as F
import pdb
import torch
from torch import nn
from metrics.clustering import kmeans


class Memory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_centroid = config.num_centroid
        self.hidden_size = config.hidden_size
        self.max_seq_length = config.max_seq_length
        #self.centroid = nn.Embedding.from_pretrained(torch.normal(0.0, 0.39, size=(self.num_centroid,
        #                                  self.max_seq_length *
        #                                  self.hidden_size)))
        self.centroid = torch.normal(-0.01, 0.39, size=(self.num_centroid,
                                                        self.hidden_size * self.max_seq_length),
                                     requires_grad=True).to('cuda')
        self.idx_dict = {}
        self.count = {}
        for i in range(self.num_centroid):
            self.idx_dict[i] = []
            self.count[i] = 0



class Student_Memory(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_centroid = config.num_centroid
        self.hidden_size = config.hidden_size
        self.max_seq_length = config.max_seq_length
        #self.centroid = torch.rand(size=(config.num_centroid, config.hidden_size * config.max_seq_length), requires_grad=True).to('cuda')
        self.centroid = torch.nn.Embedding(config.num_centroid, config.hidden_size * config.max_seq_length)
    def forward(self, input):
        idx = torch.LongTensor([i for i in range(self.num_centroid)]).to('cuda')
        new_input = input.view(-1, self.max_seq_length * self.hidden_size)
        centroid = self.centroid(idx)
        #cka_loss = self.linear_cka_loss(new_input, centroid)
        dist = torch.cdist(new_input, centroid)
        norm_input = F.normalize(new_input, dim=-1)
        norm_weight = F.normalize(centroid)
        cos = torch.mm(norm_input, norm_weight.transpose(-1, -2))

        return (centroid, dist, cos)

    def centering(self, input):
        n = input.size()[-1]
        unit = torch.ones(size=(n, n)).to('cuda')
        I = torch.eye(n).to('cuda')
        H = I - unit / n
        return torch.matmul(torch.matmul(H, input), H)

    def linear_HSIC(self, x, y):
        if x.dim() >= 3 and y.dim() >= 3:
            l_x = torch.matmul(x, x.transpose(-2, -1))
            l_y = torch.matmul(y, y.transpose(-2, -1))
            return torch.sum(torch.sum(torch.mul(self.centering(l_x), self.centering(l_y)), dim=-1), dim=-1)

        else:
            l_x = torch.matmul(x, x.transpose(-2, -1))
            l_y = torch.matmul(y, y.transpose(-2, -1))
            return torch.sum(torch.mul(self.centering(l_x), self.centering(l_y)))

    def linear_cka_loss(self, x, y):
        hsic = self.linear_HSIC(x, y)
        var1 = torch.sqrt(self.linear_HSIC(x, x))
        var2 = torch.sqrt(self.linear_HSIC(y, y))
        return -torch.log(torch.abs(torch.div(hsic, (var1 * var2))) + 1e-8)