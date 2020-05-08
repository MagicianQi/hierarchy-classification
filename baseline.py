# -*- coding: utf-8 -*-

import os
import torch
import pickle

from dataset.preprocessing import BertVec
from dataset.dataloader import BaselineDataSet

os.makedirs("./workdir/baseline", exist_ok=True)
bv = BertVec()
labels = ["股票", "体育"]

datasets_train = BaselineDataSet("./sample_data/temp.data", labels, bv)

print(datasets_train)

dataLoader_train = torch.utils.data.DataLoader(datasets_train,
                                               batch_size=2,
                                               shuffle=False,
                                               num_workers=0)

for inputs, labels in dataLoader_train:
    print(inputs.size())
    print(labels.size())


# train_data = []
#
# with open("./sample_data/news.train", "r") as f:
#     i = 0
#     for line in f.readlines():
#         i += 1
#         label, text = line.strip().split("\t")  # 可能出错
#         label_id = labels.index(label)
#         tokens, encodings = bv.encode(text)
#         print("{}\t{}\t{}\r".format(i, len(tokens), label), end="")
#         train_data.append({"tokens": tokens,
#                            "encodings": encodings,
#                            "label": label})
#
#
# print(len(train_data))
