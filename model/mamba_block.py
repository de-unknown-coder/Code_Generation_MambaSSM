import torch
import torch.nn as nn
import config

class MambaBlock(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.W_A = nn.Linear(d_model,d_model)
        self.W_B = nn.Linear(d_model, d_model)
        self.W_C = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.h0 =  torch.nn.Parameter(torch.randn(1,d_model))
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self,X):
        Z = self.norm(X)
        A = torch.sigmoid(self.W_A(Z))
        P = torch.cumprod(A, dim=1)
        Bx = self.W_B(Z)
        cumSum = torch.cumsum(Bx/(P + 1e-8),dim=1)
        batch_h0 =  self.h0.expand(Z.shape[0], -1)
        term_1 = P * batch_h0.unsqueeze(1)
        term_2 = P * cumSum
        h = term_1 + term_2
        C = self.W_C(Z)
        Y = C*h
        Y = self.dropout(Y)
        return Y+X