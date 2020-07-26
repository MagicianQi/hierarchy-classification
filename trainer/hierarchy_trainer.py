# -*- coding: utf-8 -*-

import os

import torch
from tqdm import tqdm

from dataset.preprocessing import make_hierarchy_datasets


def train_process(dataLoader, epoch, params):
    model = params['model']
    optimizer = params['optimizer']
    criterion = params['criterion']
    device = params['device']
    logger = params['logger']

    right_num = 0
    all_num = 0
    step = 0
    dataLoader = tqdm(dataLoader)
    for inputs, labels in dataLoader:
        step += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _, _, _, _ = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        compare = torch.max(outputs, 1)[1] - labels
        compare = compare.cpu().numpy().tolist()

        right_num += compare.count(0)
        all_num += len(compare)
        dataLoader.set_description("Train Epoch: {}  loss: {}".format(epoch, str(loss.item())[0:7]))
        logger.out_print("Train Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}".format(epoch, step, loss.item(),
                                                                                    right_num, all_num),
                         with_time=True)
    print("Train Epoch: {}\tacc: {}".format(epoch, str(float(right_num / all_num))[0:7]))
    logger.out_print("Train Epoch: {}  acc: {}".format(epoch, float(right_num / all_num)), with_time=True)


def val_process(dataLoader, epoch, params):
    model = params['model']
    criterion = params['criterion']
    device = params['device']
    logger = params['logger']

    right_num = 0
    all_num = 0
    step = 0
    dataLoader = tqdm(dataLoader)
    for inputs, labels in dataLoader:
        step += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs, _, _, _, _ = model(inputs)
            loss = criterion(outputs, labels)

        compare = torch.max(outputs, 1)[1] - labels
        compare = compare.cpu().numpy().tolist()

        right_num += compare.count(0)
        all_num += len(compare)
        dataLoader.set_description("Val Epoch: {}  loss: {}".format(epoch, str(loss.item())[0:7]))
        logger.out_print("Val Epoch: {}\tIter: {}\tloss: {}\tcount: {}/{}".format(epoch, step, loss.item(),
                                                                                  right_num, all_num),
                         with_time=True)
    acc = float(right_num / all_num)
    print("Val Epoch: {}  acc: {}".format(epoch, str(float(acc))[0:7]))
    logger.out_print("Val Epoch: {}\tacc: {}".format(epoch, acc), with_time=True)
    return acc


def predict(data_path, work_dir, params):
    model = params['model']
    bv = params['bv']
    device = params['device']
    num_sentences = params['num_sentences']
    num_words = params['num_words']
    num_chars = params['num_chars']
    vec_length = params['vec_length']
    result_dir = work_dir + "predict_result/"
    print("predict result path: {}".format(result_dir))

    os.makedirs(result_dir, exist_ok=True)

    data_list = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            split_data = line.strip().split("\t")
            if len(split_data) != 2:
                continue
            label, text = split_data
            data_list.append(text)
    data_list = tqdm(enumerate(data_list))
    for i, text in data_list:
        data_list.set_description("process text {}".format(i + 1))
        with open("{}{}.txt".format(result_dir, i), "w") as f:
            f.write("original:\n{}\n".format(text))
            tokens, text_encodings = bv.encode(text)
            hierarchy_tokens, hierarchy_encodings = make_hierarchy_datasets(tokens, text_encodings, num_sentences,
                                                                            num_words, num_chars, vec_length)
            inputs = torch.Tensor(hierarchy_encodings)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            _, _, word_attention_score, sentence_attention_score, para_attention_score = model(inputs)
            word_attentions = word_attention_score.cpu().detach().numpy().tolist()[0]
            sentence_attentions = sentence_attention_score.cpu().detach().numpy().tolist()[0]
            para_attentions = para_attention_score.cpu().detach().numpy().tolist()[0]

            sentence_result = []
            word_result = []
            char_result = []
            for s, sentence_tokens in enumerate(hierarchy_tokens):
                # 归一化
                true_sentence_length = len(hierarchy_tokens)
                para_attentions[0:true_sentence_length] = [x / sum(para_attentions[0:true_sentence_length]) for x in para_attentions[0:true_sentence_length]]
                sentence_score = para_attentions[s]
                sentence_result.append(["".join(["".join(x) for x in sentence_tokens]), sentence_score])
                for w, word_tokens in enumerate(sentence_tokens):
                    # 归一化
                    true_word_length = len(sentence_tokens)
                    sentence_attentions[s][0:true_word_length] = [x / sum(sentence_attentions[s][0:true_word_length]) for x in sentence_attentions[s][0:true_word_length]]
                    word_score = sentence_attentions[s][w]
                    word_result.append(["".join(word_tokens), sentence_score * word_score])
                    for c, char_token in enumerate(word_tokens):
                        # 归一化
                        true_char_length = len(word_tokens)
                        word_attentions[s][w][0:true_char_length] = [x / sum(word_attentions[s][w][0:true_char_length]) for x in word_attentions[s][w][0:true_char_length]]
                        char_score = word_attentions[s][w][c]
                        char_result.append([char_token, sentence_score * word_score * char_score])

            f.write("trained:\n{}\n".format("".join([x[0] for x in sentence_result])))

            # -------打印字权重-------
            f.write("------char weights(sorted)------\n")
            char_result = sorted(char_result, key=lambda x: x[1], reverse=True)
            for char, score in char_result:
                f.write("{}\t{}\n".format(char, score))

            # -------打印词权重-------
            f.write("------word weights(sorted)------\n")
            word_result = sorted(word_result, key=lambda x: x[1], reverse=True)
            for word, score in word_result:
                f.write("{}\t{}\n".format(word, score))

            # -------打印句子权重-------
            f.write("------sentence weights(sorted)------\n")
            sentence_result = sorted(sentence_result, key=lambda x: x[1], reverse=True)
            for sentence, score in sentence_result:
                f.write("{}\t{}\n".format(sentence, score))
