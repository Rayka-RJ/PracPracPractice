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

        self.transformer = nn.ModuleDist(dict(
            wte = nn.Embedding(config.vocal_size, config.n_embd),
            wpe = nn.Embedding(config.vocal_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))

        # language model output head
        # 把 transformer 最后的 hidden state (384维) 映射回词表大小 (65维)，预测下一个token。
        # Input token (65) → Embedding (65→384) → Transformer处理 → lm_head (384→65) → 预测概率
        self.lm_head = nn.Linear(config.n_embd, config.vocal_size, bias=False)
        
        # 权重共享 scheme
        # 让输入 embedding 和输出 projection 共用同一个权重矩阵。
        # 原本 wte 有 65×384 参数，lm_head 又有 384×65，共享后只需一份。GPT-2/3 都用这个 trick。
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # 每个 transformer block 有两个残差连接 (attention + MLP)。如果初始化太大，6层后残差路径累积会让梯度爆炸。
        # 把 c_proj（attention 和 MLP 的输出投影）的初始化标准差除以 √(2×层数)，让每层贡献变小。
        # 这是GPT-2 论文的初始化策略
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
        print(f'number of parameters: %.2fM' % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f'Cannot forward sequence if length {t}, block_size is only {self.block_size}'
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape(t)

        # forward propagation
        # 输入 token → wte + wpe → dropout → 6个Block → ln_f → lm_head → 输出 logits
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # back propagation
        # F.cross_entropy 期望输入是 (N, C) 格式，N = 样本数，C = 类别数（词表大小）
        # x 最后是(B, T, C)，经过lm_head后变成logits现在是(B, T, VOCAL_SIZE)，需要把B和T展平
        # logits.view(-1, 65) -> B*T, 65 ->把B个句子（每个T个词）变成B*T个独立猜词任务
        # view(-1) 的核心作用是：在保持数据不变的情况下，将 Tensor 展平为1D
        if targets is not None:
            # train mode
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference mode
            # 基于已经生成的词（前文），预测紧接着的一个词。
            # 因此，只需要把当前序列的最后一个向量推给 lm_head（分类层），算出下一个词在词表上的概率分布。
            # x[:, [-1], :] 保留了 Batch 维度。只取最后一个时间步 [-1]。保留了特征维度（Hidden Size）。
            # 结果形状： [B, 1, H]。使用 [-1] 而不是 -1 是为了保持三维形状，防止维度坍缩。
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_token, temperature=1.0, top_k=None):
        '''
        generate text
        idx: (B, T), 包含当前context的索引
        '''
        for _ in range(max_new_token):
            # 负号表示从末尾开始计数, 模型只根据最近的 block_size 个词预测下一个词。
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # forward 
            logits, _ = self(idx_cond)
            
            # 只取最后一个时间步
            logits = logits[:, -1, :] / temperature

            # 提取top k
            if top_k is not None:
                # 找出前 K 个最大的分数
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 把剩下的（较小的）分数全部变成负无穷
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # 应用softmax转换概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 追加到序列
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

'''
-------------------------------------------
1. shape和size
# k.shape      # 属性,返回 torch.Size([B, nh, T, hs])
# k.shape[-1]  # 64 (索引,不是方法调用)
# k.size(-1)   # 方法,返回 int (最后一维的大小)

-------------------------------------------
2. transpose
# 这两个完全一样
k.transpose(-2, -1)  # 交换倒数第2维和倒数第1维
k.transpose(-1, -2)  # 交换倒数第1维和倒数第2维

-------------------------------------------
3. causalmask
self.bias 是固定的: (1, 1, block_size, block_size) = (1, 1, 256, 256)

# 但实际输入可能只有 10 个 token
x: (B, 10, C)

# 手动实现需要切片:
mask = self.bias[:, :, :10, :10]  # 只用前 10x10 的部分

# is_causal=True 自动处理:
# PyTorch 自动创建 10x10 的 mask,不需要我们切片

-------------------------------------------
4. 特殊的残差投影缩放 (在 __init__ 里)
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
作用: 对残差连接的输出投影做特殊缩放

为什么需要?

残差连接会累积信号，N 层后,信号强度 ≈ √N 倍，提前缩小 1/√(2N),保持平衡

是否必须? ⚠️ 不必须,但对深层网络(12层+)很重要
能删吗? ⚠️ 6层模型删了影响不大,24层模型删了会不稳定

-------------------------------------------
5.1 logits.view(-1, logits.size(-1)) 为什么要传两个参数？ 

logits.size(-1)：这代表取出最后一个维度的大小，也就是词表大小（Vocab Size）。
-1：告诉 PyTorch 自动计算第一个维度。它会将 Batch Size 和 Seq Length 乘在一起。

5.2 ignore_index=-1 是什么？

当 targets 中的某个值等于 ignore_index（这里设为 -1）时
cross_entropy 在计算梯度和损失时会直接跳过它。
它产生的误差不会被计入总 Loss，也不会影响模型的参数更新。

-------------------------------------------
6.1 为什么 self 可以像函数一样调用 self()?

在 Python 中，如果一个类定义了 __call__ 方法，它的实例就可以像函数一样被调用。 
在 PyTorch 中，所有的 nn.Module（模型基类）都实现了 __call__。当你调用 self(idx_cond) 时，它实际上是在执行：
- 预处理（Hooks）。
- 用 forward 方法：执行你定义的模型前向传播逻辑。
- 后处理。

一句话总结：self(idx_cond) 等同于运行了模型，得到了预测结果 logits。

6.2 为什么 logits 要除以 temperature（温度）？

temperature 是控制模型“创造力”的开关：
原理：logits 是模型输出的原始分值。除以一个数（温度）会改变分值之间的差距。
低温度 (< 1.0)：分值差距拉大，概率分布变得“尖锐”。模型变得保守、确定，总选那个最可能的词。
高温度 (> 1.0)：分值差距缩小，概率分布变得“平坦”。模型变得奔放、随机，更容易选到奇奇怪怪的词。

6.3 采样与拼接
1). probs = F.softmax(logits, dim=-1)：将分数转化成总和为 $100\%$ 的概率分布。比如：{"学": 80%, "玩": 15%, "吃": 5%}。
2). idx_next = torch.multinomial(probs, num_samples=1)：根据概率“抽签”。
    注意，它不是直接选概率最大的，而是按比例抽。80% 概率抽到“学”，但也可能手气不好抽到“玩”。这保证了 AI 生成的多样性。
3). idx = torch.cat((idx, idx_next), dim=1)：把新抽到的词（idx_next）拼接到原来的句子（idx）后面。
    旧 idx: ["我", "爱"]idx_next: ["学"]新 idx: ["我", "爱", "学"]

idx即词元id，直接对应文本。
--------------------------------------------
7. inference全流程

假设 idx 是 [[10, 25]]（代表“我爱”），top_k=2。
模型预测：self(idx) 算出下一个字的分数。，self(idx) 得到的原始 logits 形状是 [Batch, 2, Vocab_Size]。所以要取[:,-1,:]，这是基于id25算下一个词的概率
Top-K：只看前两名，假设是“学”和“玩”。
Softmax：算出概率，“学” 占 90%，“玩” 占 10%。
采样：由于“学”概率极高，抽签抽到了“学”（ID 为 33）。
拼接：idx 变成 [[10, 25, 33]]（代表“我爱学”）。
循环：下一次循环，模型就会根据“我爱学”去预测下一个字了。

--------------------------------------------
8. 训练与推理

1). 训练模式：一次性“并行”生成（Teacher Forcing）
在训练模型时，我们其实已经有了正确答案（Target）。这时候我们不需要一个词一个词地蹦，而是利用 Transformer 的并行能力。

操作：我们将整句话 [我, 爱, 学, 习] 一次性输入模型。
逻辑：模型会同时输出：
位置 1 对位置 2 的预测
位置 2 对位置 3 的预测
位置 3 对位置 4 的预测

计算：我们一次性拿到了整句话所有位置的 logits，直接和 Target 计算 Cross Entropy Loss。

结论：训练时不需要 for 循环，效率极高。


2). 生成模式：必须“串行” (自回归)

而在 generate 函数中，因为我们没有 Target，模型必须先生成第 3 个词，才能知道第 4 个词的上下文是什么。
痛点：随着句子变长，idx 越来越大。每次 self(idx) 都要重新计算前面所有词的注意力（Attention）。

重复劳动：比如生成第 100 个词时，前 99 个词的计算在之前的循环里已经做过 99 遍了！


3). 高级优化：一次性生成的“进化版” (KV Cache)

为了让生成变得“不一样”（更快），大模型（如 GPT-4, Llama 3）在推理时会使用一种叫 KV Cache 的技术。
代码逻辑的变化： 不再是 logits, _ = self(idx_cond)，而是类似于： logits, past_key_values = self(next_token_only, past_key_values)
只传一个词：每次循环只把上一步刚生成的那个词传进去。
缓存记忆：把之前词的计算结果（Key 和 Value 矩阵）存在内存里。

结果：计算量从“每次都要算整句”变成了“每次只算新词”，速度提升了几十倍。
'''