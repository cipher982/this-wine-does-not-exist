#!/bin/sh

python3 -m pip install wandb pytorch_lightning deepspeed

git clone https://github.com/huggingface/transformers
cd transformers
git checkout 7e662e6a3be0ece4 
python3 -m pip install .

cd examples/seq2seq
python3 -m pip install -r requirements.txt
wget https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz
tar -xzvf wmt_en_ro.tar.gz