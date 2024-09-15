from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConfidenceAttention(nn.Module):
    def __init__(self, hidden_dim: int, confidence_dim: int, head_num: int = 8, dropout: float = 0.1):
        super(ConfidenceAttention, self).__init__()

        assert hidden_dim % head_num == 0, "hidden_dim must be divisible by head_num"
        assert confidence_dim % head_num == 0, "confidence_dim must be divisible by head_num"

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.confidence_dim = confidence_dim
        self.confidence_head_dim = confidence_dim // head_num

        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.proj_abs_c = nn.Linear(confidence_dim, 1)
        self.proj_rel_c = nn.Linear(confidence_dim, 1)
        self.w_concat = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.constant_(self.proj_q.bias, 0)
        nn.init.constant_(self.proj_k.bias, 0)
        nn.init.constant_(self.proj_v.bias, 0)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, abs_confidence: torch.Tensor, rel_confidence: torch.Tensor, mask: Optional[torch.Tensor] = None):
        query = self.proj_q(query)
        key = self.proj_k(key)
        value = self.proj_v(value)

        query = rearrange(query, "b n (h hd) -> (b h) n hd", h=self.head_num, hd=self.head_dim)
        key = rearrange(key, "b n (h hd) -> (b h) n hd", h=self.head_num, hd=self.head_dim)
        value = rearrange(value, "b n (h hd) -> (b h) n hd", h=self.head_num, hd=self.head_dim)
        
        rel_c = F.sigmoid(self.proj_rel_c(rel_confidence))
        abs_c = F.sigmoid(self.proj_abs_c(abs_confidence))
        confidence = rel_c * abs_c.transpose(0, 1)
        confidence = confidence.repeat(1, 1, self.head_num)
        confidence = rearrange(confidence, "b n h -> (b h) n") 

        score = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        score = score * confidence.unsqueeze(dim=1)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.head_num, 1)
            mask = rearrange(mask, "b h n -> (b h) n")
            score = score.masked_fill_(mask.unsqueeze(dim=1), -float("Inf"))

        score = F.softmax(score, dim=-1)
        score = self.dropout(score)

        out = torch.bmm(score, value)

        out = rearrange(out, "(b h) n hd -> b n (h hd)", h=self.head_num, hd=self.head_dim)
        out = self.w_concat(out)

        abs_confidence = rearrange(abs_confidence.repeat(1, abs_confidence.shape[0], 1), "b n (h hd) -> (b h) n hd", h=self.head_num, hd=self.confidence_head_dim)
        out_confidence = torch.bmm(score, abs_confidence)
        out_confidence = rearrange(out_confidence, "(b h) n hd -> b n (h hd)", h=self.head_num, hd=self.confidence_head_dim)

        return out, out_confidence