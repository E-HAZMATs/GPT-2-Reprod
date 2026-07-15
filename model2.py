import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

'''
T0D0$
TODO: Apply KV Cache?
TODO: Add dropout regging later.
'''
@dataclass
class GPTConfig:
    seq_len: int = 8
    embedding_len = 32
    n_heads = 8
    n_blocks = 12
    vocab_size = 50

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
        # > matmul softmax result (wei) with V -> project.
        B, T, C = x.size() 
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
        # XXX: Constant. Move to constructor?
        tril = torch.tril(torch.ones(T,T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        # Each vector at last dim, should sum to 1.
        affinity = F.softmax(wei, dim=-1)
        attention = affinity  @ v
        # `contiguous` sorta applies the transposition in memory. Otherwise the `view` will fail
        attention = attention.transpose(1,2).contiguous().view(B,T,C)
        # This makes each head result PER TOKEN communicate/concat.
        out = self.proj_lin(attention)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        g_approx = 'none' # or 'tanh'.  altho, gelu was solved in newer pytorch? 
        self.lin1 = nn.Linear(config.embedding_len, config.embedding_len * 4)
        self.gelu = nn.GELU(approximate=g_approx)
        self.lin2 = nn.Linear(config.embedding_len * 4, config.embedding_len)

    def forward(self, x):
        # DON'T FORGET ABOUT RES CONS.
        # in transformer diagram the output of this layer is added to residual pathway.
        # but gpt does things different, so...?
        # whole layer is a single residual block? so the addition is in the Atten+FeedFW block?
        x = self.lin1(x) # > (B, T, C * 4)
        x = self.gelu(x) # > (B, T, C * 4)
        x = self.lin2(x) # > (B,T,C)
        return x

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_len)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_len)
        self.feed_fw = FeedForward(config)
    
    def forward(self, x):
        '''
        input: (B, T, C)
        output: (B, T, C)
        '''
        x = self.ln_1(x)
        x = x + self.attn(x)
        x = self.ln_2(x)
        x = x + self.feed_fw(x)
        return x

class GPT(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            embedding_table = nn.Embedding(config.vocab_size, config.embedding_len),
            positional_embedding = nn.Embedding(config.seq_len, config.embedding_len),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)]), 
            ln = nn.LayerNorm()
        ))
        
'''
Some notes:

- GPT, differs than the transformer.png. It's a decoder-only transformer.
the LN is done I think before the attention adn FW instead of after shown in the diagram.
- attn and FW are each a residual block. their output to residual pathway.

'''

conf = GPTConfig()
print(conf.seq_len)
s_attn = SelfAttention(conf)
x = torch.rand(4, conf.seq_len, conf.embedding_len)
out = s_attn(x)
feed_fw = FeedForward(conf)
out = feed_fw(out)

block = Block(conf)
out = block(x)