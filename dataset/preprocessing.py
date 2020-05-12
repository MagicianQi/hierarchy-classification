# -*- coding: utf-8 -*-

from typing import List

import re

from bert_serving.client import BertClient


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
