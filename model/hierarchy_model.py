# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.self_attention import SelfAttention


class HierarchyModel(nn.Module):
    """
    Hierarchical Attention Model
    """

    def __init__(self, input_size, num_classes):
        super(HierarchyModel, self).__init__()
        self.word_attention_layer = SelfAttention(input_size=input_size)
        self.sentence_attention_layer = SelfAttention(input_size=input_size)
        self.para_attention_layer = SelfAttention(input_size=input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x, word_attention_score = self.word_attention_layer(x)
        x = torch.sum(x, dim=3)
        x = torch.squeeze(x, 3)

        x, sentence_attention_score = self.sentence_attention_layer(x)
        x = torch.sum(x, dim=2)
        x = torch.squeeze(x, 2)

        x, para_attention_score = self.para_attention_layer(x)
        x = torch.sum(x, dim=1)
        x = torch.squeeze(x, 1)

        logits = self.fc(x)
        y = F.softmax(logits, dim=-1)
        return y, logits, word_attention_score, sentence_attention_score, para_attention_score
