# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data


class BaselineDataSet(data.Dataset):
    def __init__(self, data_path, class_names, bert_vec, text_length, vec_length):
        fh = open(data_path, "r")
        self.classes = class_names
        self.text_length = text_length
        self.vec_length = vec_length
        self.bv = bert_vec
        data_combined = []
        for line in fh.readlines():
            split_data = line.strip().split("\t")
            if len(split_data) != 2:
                continue
            label, text = split_data
            label_id = class_names.index(label)
            data_combined.append((text, label_id))
        self.data_combined = data_combined

    def __getitem__(self, index):
        text, label_id = self.data_combined[index]
        _, encodings = self.bv.encode(text)
        if len(encodings) >= self.text_length:
            # truncation
            encodings = encodings[0:self.text_length]
        else:
            # padding
            encodings.extend([[0.0 for _ in range(self.vec_length)] for _ in range(self.text_length - len(encodings))])
        return torch.Tensor(encodings), label_id

    def __len__(self):
        return len(self.data_combined)

    def getName(self):
        return self.classes
