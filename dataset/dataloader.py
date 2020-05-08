# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data


class BaselineDataSet(data.Dataset):
    def __init__(self, data_path, class_names, bert_vec):
        fh = open(data_path, "r")
        self.classes = class_names
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
        tokens, encodings = self.bv.encode(text)
        label = [0 for _ in range(len(self.classes))]
        label[label_id] = 1
        encodings = encodings[0:10]
        return torch.Tensor(encodings), torch.Tensor(label)

    def __len__(self):
        return len(self.data_combined)

    def getName(self):
        return self.classes
