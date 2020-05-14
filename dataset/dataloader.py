# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data

from utils.seg_tools import cut_paragraph, cut_sentence


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

        # ------------------切分段落------------------
        hierarchy_encodings = []
        sentence_cuts = cut_paragraph("".join(tokens))
        encoding_index = 0  # 由于句子中存在'[UNK]'等， 所以采用index方式，按字符截取encodings不行
        for s_index, sentence in enumerate(sentence_cuts):
            # 中间存储
            hierarchy_encodings_temp = []

            # ------------------切分句子------------------
            word_cuts = cut_sentence(sentence)
            word_cuts_lens = []
            for word in word_cuts:
                if word in tokens:
                    word_cuts_lens.append(1)
                else:
                    word_cuts_lens.append(len(word))
            word_slices = [slice(encoding_index + sum(word_cuts_lens[0:i]), encoding_index + sum(word_cuts_lens[0:i]) + x)
                           for i, x in enumerate(word_cuts_lens)]

            for w_index, word in enumerate(word_cuts):
                word_encodings = text_encodings[word_slices[w_index]]

                # ------------------词的截断和padding------------------
                # 判断是否'[UNK]'等， 如果是，那么词的长度为1
                if word in tokens:
                    word_encodings.extend(self._make_zero_tensors([self.num_chars - len(word_encodings)]))
                else:
                    if len(word) >= self.num_chars:
                        word_encodings = word_encodings[0:self.num_chars]
                    else:
                        word_encodings.extend(self._make_zero_tensors([self.num_chars - len(word_encodings)]))
                hierarchy_encodings_temp.append(word_encodings)

            encoding_index += sum(word_cuts_lens)  # index变为下一个句子的起点

            # ------------------句子的截断和padding------------------
            if len(hierarchy_encodings_temp) >= self.num_words:
                hierarchy_encodings_temp = hierarchy_encodings_temp[0:self.num_words]
            else:
                hierarchy_encodings_temp.extend(
                    self._make_zero_tensors([self.num_words - len(hierarchy_encodings_temp), self.num_chars]))
            hierarchy_encodings.append(hierarchy_encodings_temp)

        # ------------------段落的截断和padding------------------
        if len(hierarchy_encodings) >= self.num_sentences:
            hierarchy_encodings = hierarchy_encodings[0:self.num_sentences]
        else:
            hierarchy_encodings.extend(
                self._make_zero_tensors([self.num_sentences - len(hierarchy_encodings), self.num_words, self.num_chars]))

        return torch.Tensor(hierarchy_encodings), label_id

    def __len__(self):
        return len(self.data_combined)

    def _make_zero_tensors(self, dimensions):
        zero_vector = [0.0 for _ in range(self.vec_length)]
        if len(dimensions) == 1:
            return [zero_vector for _ in range(dimensions[0])]
        else:
            for i in range(len(dimensions)):
                return [self._make_zero_tensors(dimensions[i+1:]) for _ in range(dimensions[i])]
