# -*- coding: utf-8 -*-

import os

import torch

from utils.seg_tools import cut_sentence, cut_paragraph


def train_process(dataLoader, epoch, params):
    model = params['model']
    optimizer = params['optimizer']
    criterion = params['criterion']
    device = params['device']
    logger = params['logger']

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
        print("Train Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}".format(epoch, step, loss.item(), right_num, all_num))
        logger.out_print("Train Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}\n".format(epoch, step, loss.item(),
                                                                                      right_num, all_num),
                         with_time=True)
    print("Train Epoch: {}\tacc: {}".format(epoch, float(right_num / all_num)))
    logger.out_print("Train Epoch: {}\tacc: {}\n".format(epoch, float(right_num / all_num)), with_time=True)


def val_process(dataLoader, epoch, params):
    model = params['model']
    criterion = params['criterion']
    device = params['device']
    logger = params['logger']

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
        print("Val Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}".format(epoch, step, loss.item(), right_num, all_num))
        logger.out_print("Val Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}\n".format(epoch, step, loss.item(),
                                                                                    right_num, all_num),
                         with_time=True)
    acc = float(right_num / all_num)
    print("Val Epoch: {}\tacc: {}".format(epoch, acc))
    logger.out_print("Val Epoch: {}\tacc: {}\n".format(epoch, acc), with_time=True)
    return acc


def predict(data_path, work_dir, params):
    model = params['model']
    text_length = params['text_length']
    vec_length = params['vec_length']
    bv = params['bv']
    device = params['device']
    result_dir = work_dir + "predict_result/"

    os.makedirs(result_dir, exist_ok=True)

    data_list = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            split_data = line.strip().split("\t")
            if len(split_data) != 2:
                continue
            label, text = split_data
            data_list.append(text)

    for i, text in enumerate(data_list):
        print("process text {}".format(i))
        with open("{}{}.txt".format(result_dir, i), "w") as f:
            f.write("original:\n{}\n".format(text))
            tokens, encodings = bv.encode(text)
            if len(encodings) >= text_length:
                tokens = tokens[0:text_length]
                encodings = encodings[0:text_length]
            else:
                encodings.extend([[0.0 for _ in range(vec_length)] for _ in range(text_length - len(encodings))])

            f.write("trained:\n{}\n".format("".join(tokens)))
            inputs = torch.Tensor(encodings)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            _, _, attentions = model(inputs)
            attentions = attentions.cpu().detach().numpy().tolist()[0]

            f.write("------char weights(sorted)------\n")
            char_weights = [[t, a] for t, a in zip(tokens, attentions[0:len(tokens)])]
            char_weights = sorted(char_weights, key=lambda x: x[1], reverse=True)
            for token, attention in char_weights:
                f.write("{}\t{}\n".format(token, attention))

            f.write("------word weights(sorted)------\n")

            word_cuts = cut_sentence("".join(tokens))
            word_cuts_lens = []
            for word in word_cuts:
                if word in tokens:
                    word_cuts_lens.append(1)
                else:
                    word_cuts_lens.append(len(word))
            word_attention_slices = [slice(sum(word_cuts_lens[0:i]), sum(word_cuts_lens[0:i]) + x)
                                     for i, x in enumerate(word_cuts_lens)]
            word_attentions = [sum(attentions[x]) for x in word_attention_slices]
            word_weights = [[t, a] for t, a in zip(word_cuts, word_attentions)]
            word_weights = sorted(word_weights, key=lambda x: x[1], reverse=True)
            for token, attention in word_weights:
                f.write("{}\t{}\n".format(token, attention))

            f.write("------sentence weights(sorted)------\n")
            sentence_cuts = cut_paragraph("".join(tokens))
            sentence_cuts_lens = list(map(len, sentence_cuts))
            sentence_attention_slices = [slice(sum(sentence_cuts_lens[0:i]), sum(sentence_cuts_lens[0:i]) + x)
                                         for i, x in enumerate(sentence_cuts_lens)]
            sentence_attentions = [sum(attentions[x]) for x in sentence_attention_slices]
            sentence_weights = [[t, a] for t, a in zip(sentence_cuts, sentence_attentions)]
            sentence_weights = sorted(sentence_weights, key=lambda x: x[1], reverse=True)
            for token, attention in sentence_weights:
                f.write("{}\t{}\n".format(token, attention))
