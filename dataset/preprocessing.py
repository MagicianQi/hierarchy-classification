# -*- coding: utf-8 -*-

from typing import List

import re
from bert_serving.client import BertClient

from utils.seg_tools import cut_sentence, cut_paragraph


def make_hierarchy_datasets(tokens, text_encodings, num_sentences, num_words, num_chars, vec_length):
    # ------------------切分段落------------------
    hierarchy_tokens = []
    hierarchy_encodings = []
    sentence_cuts = cut_paragraph("".join(tokens))
    encoding_index = 0  # 由于句子中存在'[UNK]'等， 所以采用index方式，按字符截取encodings不行
    for s_index, sentence in enumerate(sentence_cuts):
        # 中间存储
        hierarchy_tokens_temp = []
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
                word_tokens = [word]
                word_encodings.extend(make_zero_tensors([num_chars - len(word_encodings)], vec_length))
            else:

                word_tokens = [s.strip() for s in word]
                if len(word) >= num_chars:
                    word_encodings = word_encodings[0:num_chars]
                    word_tokens = word_tokens[0:num_chars]
                else:
                    word_encodings.extend(make_zero_tensors([num_chars - len(word_encodings)], vec_length))

            hierarchy_tokens_temp.append(word_tokens)
            hierarchy_encodings_temp.append(word_encodings)

        encoding_index += sum(word_cuts_lens)  # index变为下一个句子的起点

        # ------------------句子的截断和padding------------------
        if len(hierarchy_encodings_temp) >= num_words:
            hierarchy_encodings_temp = hierarchy_encodings_temp[0:num_words]
            hierarchy_tokens_temp = hierarchy_tokens_temp[0:num_words]
        else:
            hierarchy_encodings_temp.extend(
                make_zero_tensors([num_words - len(hierarchy_encodings_temp), num_chars], vec_length))
        hierarchy_tokens.append(hierarchy_tokens_temp)
        hierarchy_encodings.append(hierarchy_encodings_temp)

    # ------------------段落的截断和padding------------------
    if len(hierarchy_encodings) >= num_sentences:
        hierarchy_encodings = hierarchy_encodings[0:num_sentences]
        hierarchy_tokens = hierarchy_tokens[0:num_sentences]
    else:
        hierarchy_encodings.extend(
            make_zero_tensors([num_sentences - len(hierarchy_encodings), num_words, num_chars], vec_length))

    return hierarchy_tokens, hierarchy_encodings


def make_zero_tensors(dimensions, vec_length):
    zero_vector = [0.0 for _ in range(vec_length)]
    if len(dimensions) == 1:
        return [zero_vector for _ in range(dimensions[0])]
    else:
        for i in range(len(dimensions)):
            return [make_zero_tensors(dimensions[i + 1:], vec_length) for _ in range(dimensions[i])]


class BertVec(object):
    """BertVec
        Get word vector from bert。
        Model is from: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    """

    def __init__(self, max_seq_len=512):
        self.bc = BertClient()
        self.max_seq_len = max_seq_len

    def encode(self, text: str) -> (List, List):
        """
        Get the encoding of each word of the target text
        :param text: Target text
        :return:
            tokens: List of characters
            encodings: List of encodings
        """
        text = self.text_process(text)
        if len(text) > (self.max_seq_len - 2):
            result = [[t, e] for t, e in map(self._per, self.cut_text(text, self.max_seq_len - 2))]
            tokens = []
            encodings = []
            for token, encoding in result:
                tokens.extend(token)
                encodings.extend(encoding)
            return tokens, encodings
        else:
            return self._per(text)

    def _per(self, text: str) -> (List, List):
        """
        Request bert service
        """
        result = self.bc.encode([text], show_tokens=True)
        embeddings, tokens = result

        embeddings = embeddings.tolist()[0]
        tokens = tokens[0]
        num_tokens = len(tokens)

        # 去掉[PAD]
        embeddings = embeddings[0:num_tokens]

        # 去掉[CLS]
        embeddings.pop(0)
        tokens.pop(0)

        # 去掉[SEP]
        embeddings.pop(-1)
        tokens.pop(-1)

        return tokens, embeddings

    @staticmethod
    def text_process(text: str) -> str:
        """
        Text specification, remove blank space etc.
        """
        result = ""
        text_list = [s.strip() for s in text]
        for each in text_list:
            if each != "":
                result += each
        return result

    @staticmethod
    def cut_text(text: str, slice_len: int) -> List:
        """
        Cut string by fixed length
        """
        text_list = re.findall('.{' + str(slice_len) + '}', text)
        text_list.append(text[(len(text_list) * slice_len):])
        if text_list[-1] == "":
            return text_list[:-1]
        return text_list


if __name__ == "__main__":
    pass
