# -*- coding: utf-8 -*-

"""
start bert server
1:
sudo docker build -t bert-as-service -f ./docker/Dockerfile .
2:
sudo docker run \
--runtime nvidia \
--name bert-server-docker \
--restart always \
-e CUDA_VISIBLE_DEVICES=0 \
-dit \
-p 5555:5555 \
-p 5556:5556 \
-v /home/qishuo/PycharmProjects/hierarchy-classification/models/chinese_L-12_H-768_A-12:/model \
-t bert-as-service 1
"""

from typing import List

import re

from bert_serving.client import BertClient


class BertVec:

    def __init__(self):
        self.bc = BertClient()
        self.max_seq_len = 512

    def encode(self, text: str) -> (List, List):
        text = text.strip().replace(" ", "")
        if len(text) > (self.max_seq_len - 2):
            result = [[t, e] for t, e in map(self._per, self.cut_text(text, self.max_seq_len - 2))]
            tokens = []
            embeddings = []
            for token, embedding in result:
                tokens.extend(token)
                embeddings.extend(embedding)
            return tokens, embeddings
        else:
            return self._per(text)

    def _per(self, text: str) -> (List, List):
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
    def cut_text(text, slice_len):
        text_list = re.findall('.{' + str(slice_len) + '}', text)
        text_list.append(text[(len(text_list) * slice_len):])
        return text_list


if __name__ == "__main__":
    bv = BertVec()
    bv_tokens, bv_embeddings = bv.encode("今天天气不错！")
