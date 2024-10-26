# 看尾节点
# 遍历节点，如果有入度，则找到所有的入度的边，建立该节点的子图，进行不同关系的聚合
# query-driven ?
# 更新
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Linear, Softmax
from torch.nn import Parameter as Param
import math
import torch_geometric.backend
import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.utils import index_sort, one_hot, scatter, spmm
from torch_geometric.utils.sparse import index2ptr


class QueryDrivenGNNNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'mean',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self._use_segment_matmul_heuristic_output: Optional[bool] = None

        if num_bases is not None:
            self.weight = Parameter(
                torch.empty(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.empty(num_relations, num_blocks,
                            in_channels[0] // num_blocks,
                            out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.empty(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.empty(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        #self.q_proj = Linear(4096, self.in_channels)
        self.q_proj = Linear(4096, self.in_channels)
        self.k_proj = Linear(self.in_channels, self.in_channels)
        self.v_proj = Linear(self.in_channels, self.in_channels)
        self.softmax = Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None, query = None):
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_l = x_l.to(torch.float32)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]
        
        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)
        
        in_node = list(set(edge_index[0].tolist()))
        for node in in_node[::-1]:# 需要更新的
            replaced_idx = torch.nonzero(edge_index[0] == node)
            from_node = edge_index[1][replaced_idx[:,0]]
            q = self.q_proj(query.to(self.q_proj.weight.device))
            k = self.k_proj(x_l[from_node])
            v = self.v_proj(x_l[from_node])
            #score = self.softmax(torch.cosine_similarity(q,k, dim = 1))
            #from_node = from_node[score.topk(2).indices]
            edge_type_tmp = edge_type[replaced_idx[:,0]]
            v = torch.matmul(v.unsqueeze(1), weight[edge_type_tmp]).squeeze(1)
            #k = torch.matmul(k.unsqueeze(1), weight[edge_type_tmp]).squeeze(1)
            #edge_type_tmp = edge_type[from_node]
            message_progate = torch.matmul(self.softmax(torch.matmul(q,k.transpose(-1,-2))/ math.sqrt(q.shape[1])), v)
            #message_progate = torch.mean(x_l[from_node].unsqueeze(1) @ weight[edge_type_tmp], dim = 0).squeeze(0)
            out[node] = out[node] + message_progate
            
        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out
            
