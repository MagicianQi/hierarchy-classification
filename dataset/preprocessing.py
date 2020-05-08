# -*- coding: utf-8 -*-

from typing import List

import re

import jieba
from bert_serving.client import BertClient


def cut_paragraph(para: str) -> List:
    """
    Paragraph segmentation.
    Code is from: https://blog.csdn.net/blmoistawinde/article/details/82379256
    """
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def cut_sentence(sentence: str) -> List:
    """
    Sentence segmentation.
    """
    return [x for x in jieba.cut("".join(sentence), cut_all=False)]


class BertVec(object):
    """BertVec
        Get word vector from bert。
        Model is from: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    """

    def __init__(self):
        self.bc = BertClient()
        self.max_seq_len = 512

    def encode(self, text: str) -> (List, List):
        """
        Get the encoding of each word of the target text
        :param text: Target text
        :return:
            tokens: List of characters
            encodings: List of encodings
        """
        text = text.strip().replace(" ", "").replace("　", "")
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
    def cut_text(text: str, slice_len: int) -> List:
        """
        Cut string by fixed length
        """
        text_list = re.findall('.{' + str(slice_len) + '}', text)
        text_list.append(text[(len(text_list) * slice_len):])
        return text_list


if __name__ == "__main__":
    bv = BertVec()
    bv_tokens, bv_embeddings = bv.encode("美邦服饰数据显示。")
    print(len(bv_tokens))
