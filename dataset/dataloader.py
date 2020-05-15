# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data

from dataset.preprocessing import make_hierarchy_datasets


class BaselineDataSet(data.Dataset):
    def __init__(self, data_path, class_names, bert_vec, max_text_length, vec_length):
        self.classes = class_names
        self.max_text_length = max_text_length
        self.vec_length = vec_length
        self.bv = bert_vec
        data_combined = []
        with open(data_path, "r") as fh:
            for line in fh.readlines():
                split_data = line.strip().split("\t")
                # Invalid format
                if len(split_data) != 2:
                    continue
                label, text = split_data
                label_id = class_names.index(label)
                data_combined.append((text, label_id))
        self.data_combined = data_combined

    def __getitem__(self, index):
        text, label_id = self.data_combined[index]
        # 此处造成死锁 原因为getitem函数引用了第三方库
        # 若把这个函数放到init中，内存估计不足。
        _, encodings = self.bv.encode(text)
        if len(encodings) >= self.max_text_length:
            # truncation
            encodings = encodings[0:self.max_text_length]
        else:
            # padding
            encodings.extend(
                [[0.0 for _ in range(self.vec_length)] for _ in range(self.max_text_length - len(encodings))])
        return torch.Tensor(encodings), label_id

    def __len__(self):
        return len(self.data_combined)


class HierarchyDataSet(data.Dataset):

    def __init__(self, data_path, class_names, bert_vec, vec_length, num_sentences, num_words,
                 num_chars):
        self.classes = class_names
        self.vec_length = vec_length
        self.bv = bert_vec
        self.num_sentences = num_sentences
        self.num_words = num_words
        self.num_chars = num_chars
        data_combined = []
        with open(data_path, "r") as fh:
            for line in fh.readlines():
                split_data = line.strip().split("\t")
                # Invalid format
                if len(split_data) != 2:
                    continue
                label, text = split_data
                label_id = class_names.index(label)
                data_combined.append((text, label_id))
        self.data_combined = data_combined

    def __getitem__(self, index):
        text, label_id = self.data_combined[index]
        tokens, text_encodings = self.bv.encode(text)

        _, hierarchy_encodings = make_hierarchy_datasets(tokens, text_encodings, self.num_sentences,
                                                         self.num_words, self.num_chars, self.vec_length)

        return torch.Tensor(hierarchy_encodings), label_id

    def __len__(self):
        return len(self.data_combined)
