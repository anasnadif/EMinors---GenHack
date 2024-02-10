import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import math


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

def linear_beta_schedule(timesteps, start=0.0001, end=0.01):
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def forward_diffusion(x_0, t):
    # Generate noise tensor
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[:, None][t]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[:, None][t]
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Generate beta schedule
T = 10000
betas = linear_beta_schedule(T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def sample(reverse_model, noise=None, n_samples=20, dim=4):
  reverse_model.eval()
  with torch.no_grad():
    if noise is not None:
      n_samples = noise.shape[0]
      x_t = torch.tensor(noise[:,:dim])
    else :
      x_t = torch.randn(n_samples, dim)
    for t in range(T-1, -1, -1):
        z = torch.randn(n_samples, dim)
        term = sqrt_recip_alphas[t] * (x_t - betas[t] / sqrt_one_minus_alphas_cumprod[t] * reverse_model(x_t, torch.full((n_samples,), t)))
        if t > 0:
            x_t = term + torch.sqrt(betas[t]) * z
        else:
            x_t = term
  reverse_model.train()
  return x_t.to("cpu").numpy()



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.reshape(-1, self.dim)
        
class ReverseNoise(nn.Module):
  def __init__(self, hidden_size = 100, dim=4, time_emb_dim=30, num_classes=None):
    super(ReverseNoise, self).__init__()


    self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU(),
            )
    self.dense_layers = nn.Sequential(
                       nn.Linear(dim+time_emb_dim, hidden_size),
                       nn.Sigmoid(),
                       nn.Linear(hidden_size, hidden_size),
                       nn.Tanh(),
                       nn.Linear(hidden_size, dim)
                      )
  def forward(self, x, t, y=None):
    t = t.reshape(-1,1)
    time = self.time_mlp(t)
    input = torch.cat((x, time), dim=1)
    output = self.dense_layers(input)
    #time_embedded = self.time_mlp()
    return output
  



