# -*- coding: utf-8 -*-

from typing import List

import re

import jieba

jieba.load_userdict('./static/jieba_custom_word_dict.txt')


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
    return [x for x in jieba.cut(sentence, cut_all=False)]


# def load_jieba_custom_words():
#     words = []
#     with open('./static/jieba_custom_word_dict.txt', "r") as f:
#         for line in f.readlines():
#             words.append(line.strip())
#     return words
