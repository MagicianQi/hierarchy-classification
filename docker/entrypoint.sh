#!/bin/sh
bert-serving-start -num_worker=$1 -model_dir /model -show_tokens_to_client -pooling_strategy NONE -max_seq_len 512