import torch
import torch.optim as optim
from model.mamba_lm import MambaLM

vocab_size = 50257
d_model = 256
n_layers = 6
model = MambaLM(vocab_size, d_model, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20
dropout_rate = 0.3