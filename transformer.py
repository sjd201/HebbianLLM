import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
from collections import Counter
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
import sys
import math
import sentencepiece as spm
import os


# SentencePiece tokenizer training
def train_tokenizer(text, model_prefix='m', vocab_size=5000):
    with open('text.txt', 'w') as f:
        f.write(text)
    spm.SentencePieceTrainer.Train(f'--input=text.txt --model_prefix={model_prefix} --vocab_size={vocab_size}')
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp

class TLM(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, num_heads):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_codes = nn.Embedding(block_size, n_embed)
        self.selft_attention_heads = MultiHead(num_heads, n_embed, block_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.FFN = nn.Sequential(nn.Linear(num_heads* n_embed, n_embed), nn.ReLU())
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embed = n_embed

    def forward(self, idx, targs=None):

        B,T = idx.shape

        emb = self.embeddings(idx)

        pos_emb = self.position_codes(torch.arange(T, device=device))
        x = emb + pos_emb
        x = self.selft_attention_heads(x)
        x = self.FFN(x) + emb
        logits = self.lm_head(x)
        
        if targs is None:
            return logits
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targs = targs.view(B*T)
            loss = F.cross_entropy(logits, targs) # -log p(targ | idx)
            return logits, loss
    
    def generate(self, idx, max_new_toks):
        #idx has shape (B, T)
        for i in range(max_new_toks):
            idx = idx[:, -self.block_size:]
            logits = self.forward(idx)  
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class Head(nn.Module):
    def __init__(self, head_size, block_size):
        super().__init__()
        self.K = nn.Linear(head_size, head_size, bias=False)
        self.Q = nn.Linear(head_size, head_size, bias=False)
        self.V = nn.Linear(head_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular mask
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.K(x) # (B, T, C)
        q = self.Q(x) # (B, T, C)
        #binding weights
        bweights = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) = (B, T, T)
        bweights = bweights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        bweights = F.softmax(bweights, dim=-1) # (B, T, T)
        v = self.V(x) # (B, T, C)
        bindings = bweights @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return bindings

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size) for _ in range(num_heads)])


    def forward(self, x):
        return torch.cat([head(x) + x for head in self.heads], dim=-1)

# Text corpus to be used
text_corpus = open("data.txt", "r").read()

# Initialize a SentencePieceProcessor object
sp = spm.SentencePieceProcessor()

# The filename of your previously trained SentencePiece model
model_filename = 'm.model'

# Load the model file into the SentencePieceProcessor
sp.Load(model_filename)
V = len(sp)

toks = sp.encode(text_corpus)


#vocab = list(set(toks))
#vocab = np.array(vocab)
#V = len(vocab)
#I = {vocab[i]:i for i in range(V)}


@torch.no_grad()
def estimate_loss(model, train_dat, val_dat, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            rand_offsets = torch.randint(len(train_dat) - block_size, (batch_size, ))                                  
            x_batch = torch.stack([train_dat[idx:idx+block_size] for idx in rand_offsets]).to(device)
            y_batch = torch.stack([train_dat[idx+1:idx+block_size+1] for idx in rand_offsets]).to(device)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#prep training data
block_size = 2 
batch_size = 1
num_heads = 1

#x = toks[:block_size]
#y = toks[1:block_size+1]
#for t in range(block_size):
#    context = x[:t+1]
#    target  = y[t]
#    print("p({} | {})".format(target, ', '.join(context)))


torch.manual_seed(1337)
data = torch.tensor([toks[i] for i in range(len(toks))])
n = int(0.75*len(data))
train_dat = data[:n]
val_dat = data[n:]
n_embed = 128
offsets = range(len(data) - block_size)
x_batch = torch.stack([data[idx:idx+block_size] for idx in offsets]).to(device)
y_batch = torch.stack([data[idx+1:idx+block_size+1] for idx in offsets]).to(device)
#logits, loss = lm(x_batch, y_batch)


#resp = lm.generate(torch.zeros((1,1), dtype=torch.long).to(device), max_new_toks=100)[0].detach().cpu().numpy()
#resp_str = ' '.join([vocab[i] for i in resp])

trainSimple = True
if trainSimple:
    lm = TLM(V, n_embed, block_size, num_heads).to(device)
    optimizer = torch.optim.Adam(lm.parameters(), lr=0.01) 
    for epoch in range(1000):
        losses = []
        for iteration in range(x_batch.shape[0]):
            optimizer.zero_grad()
            logits, loss = lm(x_batch[iteration:(iteration+1)], y_batch[iteration:(iteration+1)])
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        #losses = estimate_loss(lm, train_dat, val_dat)
        #print("epoch: {}, loss: {}".format(epoch, losses))
        #print (losses)
        print (f"Epoch: {epoch} Loss: {np.mean(losses)}", flush=True)

    for i in range(50):
        idx = x_batch[i:i+block_size,:]
        logits = lm.forward(idx)
        logits = logits[:, -1, :] # (B, C)
        probs = F.softmax(logits, dim=-1) # (B, C)
        idx_next = torch.multinomial(probs, 1) # (B, 1)
        print (sp.IdToPiece(int(y_batch[i, -1])), sp.IdToPiece(int(idx_next[0, -1])))
    
