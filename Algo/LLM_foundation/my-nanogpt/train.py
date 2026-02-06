'''
GPT - Training Script - Shakespeare token
'''
import os
import time
import math
import pickle
import numpy as np
import torch
from model import GPT, GPTConfig

#-----------------------------------------------------------
# CONFIG
#-----------------------------------------------------------
# DATA
batch_size = 64             # 批次大小
block_size = 256            # 上下文长度

# MODEL (baby GPT)
n_layer = 6                 # Transformer 层数
n_head = 6                  # 注意力头数
n_embd = 384                # embeding 
dropout = 0.2               # dropout
bias = False                # LayerNorm 和 FFN 是否使用bias

# OPTIMIZER -- Adam
learning_rate = 1e-3        # 学习率
max_iters = 5000            # 总训练步数
weight_decay = 1e-1         # AdamW 权重衰减
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0             # 梯度裁剪

# LEARNING RATE
decay_lr = True             # 是否使用学习率衰减
warmup_iters = 100          # warmup 步数
lr_decay_iters = 5000       # 应该 = max_iters
min_lr = 1e-4               # 最小学习率 = learning_rate/10

# SYSTEM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = True

# EVAL
eval_interval = 500         # 评估间隔
eval_iters = 200            # 评估时的迭代次数
log_interval = 10           # log 打印间隔
#-----------------------------------------------------------

