from torch_geometric.nn import RGCNConv, RGATConv, HGTConv
import torch
import torch.nn.functional as F
from query_driven_gnn import QueryDrivenGNNNConv
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.convs.append(QueryDrivenGNNNConv(in_channels, hidden_channels, num_relations, num_bases = 10))
        self.convs.append(QueryDrivenGNNNConv(hidden_channels, out_channels, num_relations, num_bases = 10))
        #self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases = 10))
        #for i in range(n_layers - 2):
        #    self.convs.append(QueryDrivenGNNNConv(hidden_channels, hidden_channels, num_relations, num_bases = 10))
            #self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases = 10))
        #self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases = 10))
        #self.convs.append(QueryDrivenGNNNConv(hidden_channels, out_channels, num_relations, num_bases = 10))
      
    def forward(self, x, edge_index, edge_type, query=None):
        #for conv, norm in zip(self.convs, self.norms):
        for conv in self.convs:
            x = conv(x, edge_index, edge_type, query)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x