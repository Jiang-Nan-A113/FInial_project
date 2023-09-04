import transformers
import torch
from math import sqrt
from torch import nn    # 是torch中构建神经网络的模块
from transformers import AutoConfig    # 这是一个通用配置类
from transformers import AutoTokenizer     # 这是一个通用分词器类
import torch.nn.functional as F

# model = "bert-base-uncased" #
# tokenizer = AutoTokenizer.from_pretrained(model)
# # from_pretrained() 这个方法则一气完成了模型类别推理、模型文件列表映射、模型文件下载及缓存、类对象构建等一系列操作。,
# # AutoTokenizer是通用封装，根据载入预训练模型来自适应。
# # transformers加不加没什么区别
#
# text = "time flies like an arrow"
# inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
# input1 = tokenizer("We are very happy to show you the Transformers library.", return_tensors="pt")
# #   指定返回数据类型，pt：pytorch的张量，tf：TensorFlow的张量
# #   add_special_tokens=False代表加入特殊的token
# # print(inputs.input_ids)
# # print(input1.input_ids)
#
# config = AutoConfig.from_pretrained(model)
# #   导入预训练的模型
# token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
# #   这段代码是用于在神经网络中创建一个嵌入层 (Embedding Layer)
# #   config.vocab_size: 这是词汇表的大小，即不同单词的数量。嵌入层将为每个单词创建一个唯一的向量。
# #   config.hidden_size: 这是嵌入向量的维度，也可以称为嵌入空间的维度。它决定了每个单词的嵌入向量将具有多少维度。
# print(token_emb)
#
# inputs_embeds = token_emb(inputs.input_ids)
# #   token_emb代表单词的标记索引
# #   print(inputs_embeds.size())
# #   torch.Size([1, 5, 768])代表：batch_size: 1, sequence_size: 5, 维度：768
# #   就是一次处理一句，一句长度5，一个单词有768个维度（值）
# print(inputs_embeds)
#
# Q = K = V = inputs_embeds
# #   Q是查询矩阵，其形状为 (batch_size, num_queries, dim_q)
# #   K 是键矩阵，其形状为 (batch_size, num_inputs, dim_k)，其中 num_inputs 表示输入序列的长度
# dim_k = K.size(-1)
# # print(dim_k)
# scores = torch.bmm(Q, K.transpose(1, 2) / sqrt(dim_k))
# #   K.transpose(1, 2) 是将 K 的维度从 (batch_size, num_inputs, dim_k) 转置为 (batch_size, dim_k, num_inputs)，
# #   这样可以将键的维度与查询的维度匹配
# print(scores.shape)
# wights = F.softmax(scores, dim=-1)
# #   在最后一个维度计算，保持代码可读和保持数据的整体结构不变
# print(wights.sum(dim=-1))
# #   SUM是求和的意思，以检查是否归一
# print(wights.shape, V.shape)
# attn_outputs = torch.bmm(wights, V)
# print(attn_outputs.shape)


def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2) / sqrt(dim_k))
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
        #   unsqueeze是增加矩阵维度的函数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
        #   将mask == 0 的地方的值变成负无穷，就是说将不需要注意的地方变成0
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)