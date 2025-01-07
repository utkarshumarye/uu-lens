import torch
import torch.nn.functional as F

def cosine_similarity(x, y):
    return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()
