import torch
import torch.optim as optim

# Configuration variables must be defined before importing models
vocab_size = 50257
d_model = 256
n_layers = 6
dropout_rate = 0.3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model.mamba_lm import MambaLM

model = MambaLM(vocab_size, d_model, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)