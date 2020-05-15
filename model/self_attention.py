# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F
import torch


class SelfAttention(nn.Module):
    """
    Self Attention for the last dimension
    """

    def __init__(self, input_size: int):
        """
        :param input_size: size of the last dimension
        """
        super(SelfAttention, self).__init__()
        self.weight_layer = nn.Linear(input_size, 1)

    def forward(self, x):
        weights = self.weight_layer(x)
        weights = torch.squeeze(weights, -1)
        attention_score = F.softmax(weights, dim=-1)
        out = torch.unsqueeze(attention_score, -1) * x
        return out, attention_score
