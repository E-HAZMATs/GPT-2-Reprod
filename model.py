from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.n_embd, self.n_head, block_size= config.n_embd, config.n_head, config.block_size

        # Since we're doing multi-headed attention in one module insead of different modules,
        # We would give equal parts of the embedding vector to each head. So we need to make sure the're divisible.
        assert self.n_embd % self.n_head == 0

        # Get qkv in one layer, later we split to get each. More efficient this way.
        self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        # This is the stencil/mask for the affinity matrix. Called bias to match gpt2 implementation.
        # register buffers allow us to add args i nthe module without having them as params (so no grads).
        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, n_head, T, Head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-1, -2)) * ( 1 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
@dataclass
class GPTConfig:
    block_size: int= 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50257
    batch_size: int = 32


class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.config.block_size, 'Example sequence is larger tha block size.'

        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x= self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    @classmethod
    def from_pretrained(cls, model_type):
        # Loading a pretrained model from hugging face.
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f'loading weights from pretrain model {model_type}')

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Remove stencil. Not a param.

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f'states dicts don\'t match.'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Conv1D weights.
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# autodetect device.
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print(f'used device: {device}')



num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
# model.eval()
model.to(device)

import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    data = f.read()
data = data[:1000]
tokens = enc.encode(data)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

logits

tokens = enc.encode('Hello, I\'m a language model,')
tokens = torch.tensor(tokens, dtype= torch.long)
# unsqueeze opposite of `squeeze`, we're adding a dim of size 1 at position 0.
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # shape = (5, 8)
x = tokens.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:

    with torch.no_grad():
        logits = model(x)
        logits =  logits[:, -1, :] # get next token of last timestep.
        probs = F.softmax(logits, dim=1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=1)

        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, : max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)