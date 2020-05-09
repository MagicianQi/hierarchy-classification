# -*- coding: utf-8 -*-

import os
import torch

from dataset.preprocessing import BertVec
from dataset.dataloader import BaselineDataSet
from model.base_model import BaseModel
from trainer.base_trainer import train_process, val_process

# --------------------global parameters-------------------

os.makedirs("./workdir/baseline", exist_ok=True)
bv = BertVec()
num_epochs = 100
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# --------------------data pipeline-------------------

labels = ["股票", "体育"]

# 部分段落会报错
datasets_train = BaselineDataSet("./sample_data/news.train", labels, bv, 512, 768)
datasets_val = BaselineDataSet("./sample_data/news.dev", labels, bv, 512, 768)

# num_workers不为0的话会死锁(待解决)
dataLoader_train = torch.utils.data.DataLoader(datasets_train,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0)

dataLoader_val = torch.utils.data.DataLoader(datasets_train,
                                             batch_size=32,
                                             shuffle=False,
                                             num_workers=0)

# --------------------model-------------------

model = BaseModel(input_size=768, num_classes=2)
model = model.to(device)
model.train(mode=True)

# --------------------loss function and optimizer-------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# --------------------train and val-------------------

for epoch in range(num_epochs):
    exp_lr_scheduler.step()
    train_process(dataLoader_train, model, optimizer, criterion, device, epoch)
    val_acc = val_process(dataLoader_train, model, criterion, device, epoch)
