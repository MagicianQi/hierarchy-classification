# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.self_attention import SelfAttention


class BaseModel(nn.Module):
    """
    Self Attention for the last dimension
    """

    def __init__(self, input_size, num_classes):
        super(BaseModel, self).__init__()
        self.attention_layer = SelfAttention(input_size=input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x, attention_score = self.attention_layer(x)
        x = torch.sum(x, dim=1)
        x = torch.squeeze(x, 1)
        logits = self.fc(x)
        y = F.softmax(logits, dim=-1)
        return y, logits, attention_score,
