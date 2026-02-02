# preprocessing data

# 下载数据的cli
# curl -LO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt 
import os
import numpy as np

current_dir = os.path.dirname(__file__)
input_file_path = os.path.join(current_dir, 'input.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f'length of dataset in char: {len(data):}')
# length of dataset in char: 1115394

# vocabulary
chars = sorted(list(set(data)))

# char --> idx
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

def encode(s):
    return [stoi[ch] for ch in s] # string --> idx

def decode(l):
    return ''.join([itos[i] for i in l]) # idx --> stirng

# train/eval split
n = len(data)
train = data[:int(n*0.9)]
eval = data[int(n*0.9):]

train_encode = encode(train)
eval_encode = encode(eval)

# export binary file
train_encode = np.array(train_encode, dtype=np.uint16)
eval_encode = np.array(eval_encode, dtype=np.uint16)
train_encode.tofile(os.path.join(current_dir,'train.bin'))
eval_encode.tofile(os.path.join(current_dir, 'eval.bin'))

# save the meta information
import pickle
meta = {
    'itos':itos,
    'stoi':stoi
}
with open(os.path.join(current_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)