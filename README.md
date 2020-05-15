# hierarchy-classification

confidential

## Environment

* Python 3.6.5 | Anaconda, Inc.
* GPU
* docker

## Data

* One line is one piece of data, Format : "$label\t$text"
* The method of constructing data set can be seen in class "run_classifier.py"-"ChiProcessor" or `head ./sample_data/news.train`

## Hyper-parameter in start_bert_service.sh

* NUM_WORKER : bert-as-service workers
* PATH_MODEL : Bert Model path

## How to use

* Get code : 
    * `git clone https://github.com/MagicianQi/bert_classification`
* Download and unzip model : 
    * `cd bert_classification && mkdir ckpts && cd ckpts`
    * `wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip && unzip chinese_L-12_H-768_A-12.zip && cd ../`
* Environment : 
    * `pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple`
* Start bert service (two ways): 
    1. docker(推荐)
        * Modify variables 'NUM_WORKER' and 'PATH_MODEL' in start_bert_service.sh
        * `bash start_bert_service.sh`
    2. background task
        * `bert-serving-start -num_worker=1 -model_dir ./ckpts/chinese_L-12_H-768_A-12 -show_tokens_to_client -pooling_strategy NONE -max_seq_len 512 &`
* Train and Test:
    * Modify global parameters in baseline.py
    * Run Baseline: `python baseline.py`
    * Run Hierarchy Model: `python hierarchical_attention.py`

* View training log、trained models、predict results
    * `workdir/`
    
## others

1.kill background task

`ps -ef | grep bert-serving-start | awk '{ print $2 }' | sudo xargs kill -9`
