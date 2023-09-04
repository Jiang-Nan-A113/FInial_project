import torch
import torch.nn.functional as F
from math import sqrt
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim) # three linear layer
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.q(key), self.q(value), query_mask, key_mask, mask
        )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask=None, key_mask=None, mask=None) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x


def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2) / sqrt(dim_k))
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
        #   unsqueeze 是增加矩阵维度的函数
        # unsqueeze is a function that increases the dimension of the matrix
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
        #   将mask == 0 的地方的值变成负无穷，就是说将不需要注意的地方变成0
        #   Make the values where mask == 0 negative infinity, i.e., make the places that don't need attention zero.
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)