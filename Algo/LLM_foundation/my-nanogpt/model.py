# model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:    
    def __init__(self, **kwargs) -> None:
        self.vocal_size = 65        # vocal_size
        self.block_size = 256       # T - sequence length
        self.n_head = 6             # H
        self.n_embd = 384           # C
        self.dropout = 0.2          # dropout rate
        self.n_layer = 6            # layer of transformer
        self.bias = False           # layerNorm & linear
        for k, v in kwargs.items():
            setattr(self, k, v)

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # q,k,v
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_embd = config.n_embd 
        self.n_head = config.n_head
        self.dropout = nn.Dropout(config.dropout)
        # Causal Mask
        # 把 mask 保存在模型里,但不当作可训练参数
        self.register_buffer('bias', torch.tril(torch.zeros(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))
    def forward(self, x):
        # X --> (B, T, C)
        B, T, C = x.size() 
        # q,k,v --> 3 * (B,T,C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1) # 按照每份XX来分，dim表示在哪个维度拆
        
        # multi-head (B,T,C) --> (B,T,H,E) --> (B, H, T, E)
        q = q.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)
        
        # attention calculation
        attn = q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf')) # masked!
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = attn @ v # (B, H, T, T) @ (B, H, T, E) --> (B, H, T, E)

        # # Flash attention 的版本
        # y = F.scaled_dot_product_attention(q,k,v,attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        # (B, H, T, E) --> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.vocal_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleList(dict(
            wte = nn.Embedding(config.vocal_size, config.n_embd),
            wpe = nn.Embedding(config.vocal_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocal_size, bias=False)
        self.apply(self._init_weights)

    def forward(self, x):
        pass

'''
1. shape和size
# k.shape      # 属性,返回 torch.Size([B, nh, T, hs])
# k.shape[-1]  # 64 (索引,不是方法调用)
# k.size(-1)   # 方法,返回 int (最后一维的大小)

2. transpose
# 这两个完全一样
k.transpose(-2, -1)  # 交换倒数第2维和倒数第1维
k.transpose(-1, -2)  # 交换倒数第1维和倒数第2维

3. causalmask
self.bias 是固定的: (1, 1, block_size, block_size) = (1, 1, 256, 256)

# 但实际输入可能只有 10 个 token
x: (B, 10, C)

# 手动实现需要切片:
mask = self.bias[:, :, :10, :10]  # 只用前 10x10 的部分

# is_causal=True 自动处理:
# PyTorch 自动创建 10x10 的 mask,不需要我们切片
'''