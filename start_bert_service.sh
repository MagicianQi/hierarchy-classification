#!/usr/bin/env bash

# variables
NUM_WORKER=1
PATH_MODEL=/home/qishuo/PycharmProjects/hierarchy-classification/ckpts/chinese_L-12_H-768_A-12

# Build bert-as-service image
sudo docker build -t bert-as-service -f ./docker/Dockerfile .

# bert-server-docker container
sudo docker run \
--gpus all \
--name bert-server-docker \
--restart always \
-e CUDA_VISIBLE_DEVICES=0 \
-dit \
-p 5555:5555 \
-p 5556:5556 \
-v ${PATH_MODEL}:/model \
-t bert-as-service ${NUM_WORKER}
