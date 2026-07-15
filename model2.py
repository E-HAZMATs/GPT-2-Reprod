import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

'''
T0D0$
TODO: Apply KV Cache?
'''
@dataclass
class GPTConfig:
    seq_len: int = 8
    embedding_len = 32
    n_heads = 8
    n_blocks = 0

class SelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.embedding_len = config.embedding_len
        # Each atten head processes an equal portion of the embedding so the numbers must be compatible.
        assert config.embedding_len % config.n_heads == 0
        self.qkv_lin = nn.Linear(config.embedding_len, config.embedding_len * 3)
        self.proj_lin = nn.Linear(config.embedding_len, config.embedding_len)

        # TODO: Add stencil/mask here? better with register buffer so it's no grad

    def forward(self, x):
        # rough steps: get QKV > matmul q with k > get mask > apply mask > softmax 
        # > matmul softmax result (wei) with V -> project? (or is that done before?) 
        B, T, C = x.size() 
        # XXX: Would C being smaller than embed_len cause compatibility issue with n_heads?
        # NO!!! C disappears after linear matmul. But wouldn't it prevent matmul cus of mismatch? (B,C) @ (n_emb, n_emb) if C != n_emb > crash
        assert C <= self.config.embedding_len, "wtf?" 
        qkv = self.qkv_lin(x) # B T EMBED_LEN*3
        q, k, v = qkv.split(self.config.embedding_len, -1) # split over emb dim
        
        # Pre process for multihead attn. 
        # For each example, for each head, for each timestep we have headsize result
        q = q.view(B, T, self.n_heads, self.embedding_len // self.n_heads).transpose(1,2) # (B, n_heads, T, head_size). Before (B, T, C). 
        k = k.view(B, T, self.n_heads, self.embedding_len // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.embedding_len // self.n_heads).transpose(1,2)
        # (B, n_heads, T, head_size) @ (B, n_heads, head_size, T) = (B, n_heads, T, T)
        wei = q @ k.transpose(-1, -2)
        with torch.no_grad(): # actually useful here?
            # Move to constructor?
            tril = torch.tril(torch.ones(T,T))
            wei = wei.masked_fill(tril == 0, float('-inf'))
            # Each vector at last dim, should sum to 1.
            affinity = F.softmax(wei, dim=-1)
        attention = affinity  @ v
        # `contiguous` sorta applies the transposition in memory. Otherwise the `view` will fail
        attention = attention.transpose(1,2).contiguous().view(B,T,C)
        out = self.proj_lin(attention)
        return out

conf = GPTConfig()
print(conf.seq_len)
s_attn = SelfAttention(conf)
x = torch.rand(4, conf.seq_len, conf.embedding_len)
out = s_attn(x)