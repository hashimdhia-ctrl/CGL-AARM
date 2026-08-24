import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAExpert(nn.Module):
    def __init__(self, hidden_dim=128, rank=4):
        super().__init__()
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, h):
        return self.classifier(h + self.B(self.A(h)))

class MetaGate(nn.Module):
    def __init__(self, hidden_dim=128, num_experts=3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_experts, hidden_dim))
        
    def forward(self, h):
        h_norm = F.normalize(h, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        return torch.mm(h_norm, p_norm.t()) # Cosine Similarity

class CGLAARM(nn.Module):
    def __init__(self, num_experts=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.experts = nn.ModuleList([LoRAExpert() for _ in range(num_experts)])
        self.meta_gate = MetaGate(num_experts)
        
    def forward(self, x):
        h = self.encoder(x)
        similarity_scores = self.meta_gate(h)
        selected_expert = torch.argmax(similarity_scores, dim=1)
        
        logits = torch.zeros(len(x), 2, device=x.device)
        for i, expert in enumerate(self.experts):
            mask = selected_expert == i
            if mask.any():
                logits[mask] = expert(h[mask])
        return logits, similarity_scores
