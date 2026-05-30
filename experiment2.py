"""
Part 2 — Parameter analysis of a Transformer encoder.

A small encoder-only Transformer (BERT-style) is defined in PyTorch
and used as a classifier. The goal of this part is NOT to train it,
but to inspect every sub-module's parameters and verify the total
parameter count.

Architecture (SmallTransformerEncoder):
    token_emb : nn.Embedding(vocab_size, d_model)
    pos_emb   : nn.Embedding(max_seq_len, d_model)
    encoder   : n_layers x TransformerEncoderLayer
                (multi-head self-attention + MLP + 2 x LayerNorm)
    fc_out    : nn.Linear(d_model, n_classes)   (classification head)

The script instantiates the model and prints, for every named
parameter, its shape and number of elements, plus the total
parameter count of the network.
"""

import torch
import torch.nn as nn

# Hyperparameters
d_model = 64
n_heads = 4 # d_k = d_v = d_model // n_heads = 16
d_ff = 128 # hidden layer of MLP
n_layers = 2 # number of Transformer layers
vocab_size = 1000
max_seq_len = 32
n_classes = 10

class SmallTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.encoder(h)
        return self.fc_out(h[:, 0, :])

model = SmallTransformerEncoder()

for name, param in model.named_parameters():
    print(f"{name:<55} {str(tuple(param.shape)):>20} "
        f"numel={param.numel()}")
    
print(f"\nNumber of parameters: "
    f"{sum(p.numel() for p in model.parameters())}")