# -*- coding: utf-8 -*-

import torch


def train_process(dataLoader, model, optimizer, criterion, device, epoch):
    right_num = 0
    all_num = 0
    step = 0
    for inputs, labels in dataLoader:
        step += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, logits, attention = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        compare = torch.max(outputs, 1)[1] - labels
        compare = compare.cpu().numpy().tolist()

        right_num += compare.count(0)
        all_num += len(compare)
        print("Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}".format(epoch, step, loss.item(), right_num, all_num))
    print("Epoch: {}\tacc: {}".format(epoch, float(right_num / all_num)))


def val_process(dataLoader, model, criterion, device, epoch):
    right_num = 0
    all_num = 0
    step = 0
    for inputs, labels in dataLoader:
        step += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs, logits, attention = model(inputs)
            loss = criterion(outputs, labels)

        compare = torch.max(outputs, 1)[1] - labels
        compare = compare.cpu().numpy().tolist()

        right_num += compare.count(0)
        all_num += len(compare)
        print("Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}".format(epoch, step, loss.item(), right_num, all_num))
    acc = float(right_num / all_num)
    print("Epoch: {}\tacc: {}".format(epoch, acc))
    return acc


def predict():
    pass