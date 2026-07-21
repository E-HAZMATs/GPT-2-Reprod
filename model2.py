import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import numpy as np
from helpers import save_ckpt, load_ckpt, get_lr
from time import time
'''
T0D0$
TODO: Apply KV Cache?
TODO: Add  regging later.
'''
@dataclass
class GPTConfig:
    seq_len: int = 128
    embedding_len: int = 768
    n_heads: int = 12
    n_blocks: int = 12
    vocab_size: int = 50257
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.proj_lin.GPT2_INIT = True

        # TODO: Add stencil/mask here? No need since using torch's attention.

    def forward(self, x):
        # rough steps: get QKV > matmul q with k > get mask > apply mask > softmax 
        # > matmul softmax result (wei) with V -> project.
        B, T, C = x.size() 
        assert C <= self.config.embedding_len, "huh?" 
        qkv = self.qkv_lin(x) # B T EMBED_LEN*3
        q, k, v = qkv.split(self.config.embedding_len, -1) # split over emb dim
        
        # Pre process for multihead attn. 
        # For each example, for each head, for each timestep we have headsize result
        q = q.view(B, T, self.n_heads, self.embedding_len // self.n_heads).transpose(1,2) # (B, n_heads, T, head_size). Before (B, T, C). 
        k = k.view(B, T, self.n_heads, self.embedding_len // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.embedding_len // self.n_heads).transpose(1,2)

        # Torch flash attention.
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        self.lin2.GPT2_INIT = True
    def forward(self, x):
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
        x = x + self.attn(x) # attn is residual block output added to residual pathway
        x = self.ln_2(x)
        x = x + self.feed_fw(x) # feed_fw is residual block output added to residual pathway
        return x

class GPT(nn.Module):
    
    def __init__(self, config:GPTConfig):
        '''
        '''
        super().__init__()
        self.config = config
        self.top_picks = 30 # Arg for torch.topk
        self.transformer = nn.ModuleDict(dict(
            embedding_table = nn.Embedding(config.vocab_size, config.embedding_len),
            positional_embedding = nn.Embedding(config.seq_len, config.embedding_len),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)]), 
            ln = nn.LayerNorm(config.embedding_len)
        ))
        self.linear_final = nn.Linear(config.embedding_len, config.vocab_size) # (B, T, vocab_size)

        # EMBEDDING AND OUTPUT LAYER WEIGHT TYING
        # Both weights have same shape btw.  
        self.transformer.embedding_table.weight = self.linear_final.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT2_INIT'):
                '''
                GPT2 Paper scales down residual layers' (layer output added to res path) weights
                
                 "A modified initialization which accounts
                for the accumulation on the residual path with model depth
                is used. We scale the weights of residual layers at initial
                ization by a factor of 1/√N where N is the number of
                residual layers."

                so last layers in attn and ffw.
                '''
                std *= (2 * self.config.n_blocks) ** -.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        # print(f'initialized weights for model: {module._get_name()}')
    
    # Recieves B sequences of T tokens. (B, T).
    def forward(self, x, y=None):
        '''
        TODO: Add block for when in inference and we're only concerned with next token of last timestep.
        '''
        B, T= x.size()
        assert T <= self.config.seq_len, f'sequence length should not exceed {self.config.seq_len}' # XXX: Should ==? otherwise mismatch with linear
        token_embeddings = self.transformer.embedding_table(x) # (B, T, C)
        positions = torch.arange(0, T).to(self.config.device) # Should use actual T sequence length from input instead of config. Otherwise, boarding in addition fails.
        pos_embeddings = self.transformer.positional_embedding(positions) # (seq_len)
        x = token_embeddings + pos_embeddings
        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln(x)
        logits = self.linear_final(x)
        # If inference, we're only cconcerned with next token for last time step.
        if not self.training and y is None:
            # XXX: Should time step dim be dropped?
            logits = logits[:,[-1], :] # [-1] brackets preserve dim. 
            T = 1 # For shaping multinomial
        
        loss = None
        # 
        # During eval, we have self.training = false. we want to measure loss for each timestep.
        # So the determinant is wether we have targets or not.
        # XXX: Check for silent but if i'm using truncated logits during eval.
        if y is not None:
            # Pytoch CE expects logits not probas. 
            # And it expects them to be B C T instead of B T C.
            assert T == x.size(1)
            loss = F.cross_entropy(logits.transpose(-1, -2), y)
        
        tokens = None
        if not self.training and y is None:
            # Sampling logic not needed during training/eval
            
            # Softmax over last dim (vocab_size) to get the probas for next token for each token in vocab
            probas = F.softmax(logits, dim=-1)

            # XXX: Need no_grad ctx?
            vals, indices = probas.topk(self.top_picks, dim=-1)
            idx = vals.view(B * T, self.top_picks).multinomial(1)
            idx = idx.view(B, T, 1) 
            tokens = indices.gather(-1, idx)
        return tokens, loss

    # Initializes the optimizer with weight decay applied to params with dims > 1
    def init_optim(self, lr=1e-3, weight_decay=1e-4):
        decay_params = []
        no_decay_params = []

        for param in self.parameters():
            if len(param.shape) > 1:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0}
        ]
        optim = torch.optim.AdamW(param_groups, lr)
        return optim
class DataLoader:
    '''
    TODO: train/eval sets? Need to find a good split ("good numbers").
    '''
    def __init__(self, data, batch_size, context_window, data_split):
        self.context_window = context_window
        self.batch_size = batch_size
        self.n = len(data)
        self.data = {
            'train': data[: int(self.n * data_split)],
            'eval': data[int(self.n * data_split):]
        }
        self.last_stop = 0
        self.capacity = (self.n // self.context_window) - 1 
    
    def construct_batch(self, i, type='train'):
        assert type in ['train', 'eval'], f"Type of batch must be either train or eval. Type given: {type}."
        x_batch = []
        y_batch = []
        for j in range(batch_size):
            end = self.last_stop + self.context_window
            x_batch.append(self.data[type][self.last_stop: end].to(dtype=torch.int32))
            y_batch.append(self.data[type][self.last_stop+1: end+1].to(dtype=torch.long)) # F.cross_ent expects type long for targets
            self.last_stop = end
        x = torch.stack(x_batch)
        y = torch.stack(y_batch)
        return x, y
    
'''
Some notes:
...
'''

def eval(loss):
        print('***EVALUATION***')
        gpt.eval()
        last_loss = loss # last training loss for checkpoint saving.
        losses = []
        last_stop_checkpoint = dataloader.last_stop
        dataloader.last_stop = 0
    
        for j in range(batches_count_eval):
            x, y = dataloader.construct_batch(j, 'eval')
            with torch.no_grad():
                _, loss = gpt(x, y)
            print(f"#{j} - Eval loss: {loss}")
            losses.append(loss.item())

        loss = sum(losses) / losses.__len__()
        dataloader.last_stop = last_stop_checkpoint
        print(f'Eval loss average: {loss}')
        gpt.train()

        save_ckpt(gpt, optim, epoch, i, round, last_loss, loss)

# region ARGS - Objects
conf = GPTConfig()
device = conf.device
with open('input.txt', 'r') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
tokenized_data = tokenizer.encode(text)
token_count = len(tokenized_data)
data = torch.tensor(tokenized_data, dtype=torch.uint16).to(device) # Vocab 50k, dtype max ~ 60k

del tokenized_data

conf = GPTConfig(vocab_size = 50304)
batch_size = 32
data_split = 0.9
context_window = conf.seq_len
batches_count_train = int(token_count * data_split) // (batch_size * context_window + 1) # How many batches per epoch. XXX: Some data loss probably happens, minor.
batches_count_eval = (token_count - int(token_count * data_split)) // (batch_size * context_window + 1)
training = True
iter = int(1e4)
lr = 1e-3
weight_decay = 1e-3
eval_every = 100
epochs = 200
warmup_steps = int((epochs * batches_count_train) * 0.15) # Warmup 15% of training run. Random num I picked :).
max_steps = 1e99 # When do we stop lr decay? 
max_lr = 6e-4
min_lr = max_lr * 0.1
load_model = False # Load checkpoint?
# endregion

# region Training 
'''
    General Notes
- Try running multiple epochs. Each epoch have the dataset permuatated somehow?
- apply warmup, lr decay.
- Grad scaling? 
TODO: Grad accumes
TODO: Frequent evals. Checkpoint at each eval.
'''
dataloader = DataLoader(data, batch_size, context_window, data_split)
gpt = GPT(conf).to(device)
# gpt = torch.compile(gpt) # Requires some C Compiler in PATH
optim = gpt.init_optim(lr, weight_decay)
if not training:
    gpt.eval()
param_count = sum([p.numel() for p in gpt.parameters() if p.requires_grad])
print(f'Param count: {param_count}')

round = 0
grad_clipped = 0 # How many times?
t1 = time()
loss = None
for epoch in range(epochs):
    dataloader.last_stop = 0
    for i in range(batches_count_train):

        if round % eval_every == 0 and round != 0:
            eval(loss)
        
        x, y = dataloader.construct_batch(i, 'train')
        tokens, loss = gpt(x,y)
        if round % 10 == 0:
            print(f'#{round} - train loss: {loss}')
        loss.backward()
        # Gradient clipping if norm of grads exceed given threshold (1). 
        norm = nn.utils.clip_grad_norm_(gpt.parameters(), 1)
        if norm > 1:
            grad_clipped += 1
            # print(f'{grad_clipped} Grads clipped . Pre clip norm = {norm:.2f}')

        # Alpha warmup & decay. 
        lr = get_lr(round, max_steps, warmup_steps, max_lr, min_lr)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        optim.step()
        optim.zero_grad()
        round += 1

# TODO: Add one last eval after training ends?
print(f'Training took: {(time() - t1) / 60} minutes.')
# endregion

# Region Sampling/Generating
# text = "Greetings, I'd like to have"
# gpt(torch.tensor(tokenizer.encode(text)).unsqueeze(0))
# def sample(text):
#     gpt.eval()
#     tokens = tokenizer.encode(text) # 8 TOKENS