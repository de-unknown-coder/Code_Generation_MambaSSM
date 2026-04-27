import torch
import torch.nn as nn
from model.mamba_block import MambaBlock

class MambaLM(nn.Module):
    def __init__(self,vocab_size, d_model, n_layers):
          super().__init__()
          self.blocks = nn.ModuleList([MambaBlock(d_model)for _ in range(n_layers)])
          self.embedding = nn.Embedding(vocab_size, d_model)
          self.W_out = nn.Linear(d_model, vocab_size)

    def forward(self,X):
        X = self.embedding(X)
        for block in self.blocks:
            X = block(X)
        Logits = self.W_out(X)
        return Logits
