# -*- coding: utf-8 -*-

from typing import List

import re
import jieba

from settings import JIEBA_CUSTOM_WORD_DICT_PATH

# Bert 有很多字是多个字符，把这些作为一个词处理
jieba.load_userdict(JIEBA_CUSTOM_WORD_DICT_PATH)


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
    return bert_UNK_process([x for x in jieba.cut(sentence, cut_all=False)])


def bert_UNK_process(cuts):
    """
    jieba分词的时候对于带标点的词没法区分，这个把[UNK]转换为一个词。
    '[', 'UNK', ']'   -> '[UNK]'
    """
    result = []
    for i in range(len(cuts)):
        if 2 <= i < len(cuts) - 2:
            if (cuts[i] == '[') and (cuts[i + 1] in ['UNK', 'PAD', 'CLS', 'SEP', 'MASK']) and (cuts[i + 2] == ']'):
                result.append("{}{}{}".format(cuts[i], cuts[i + 1], cuts[i + 2]))
            elif (cuts[i - 1] == '[') and (cuts[i] in ['UNK', 'PAD', 'CLS', 'SEP', 'MASK']) and (cuts[i + 1] == ']'):
                pass
            elif (cuts[i - 2] == '[') and (cuts[i - 1] in ['UNK', 'PAD', 'CLS', 'SEP', 'MASK']) and (cuts[i] == ']'):
                pass
            else:
                result.append(cuts[i])
        else:
            result.append(cuts[i])
    return result