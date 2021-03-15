#!/bin/sh

python3 -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install wandb pytorch_lightning mpi4py deepspeed

git clone https://github.com/huggingface/transformers
cd transformers
git checkout 7e662e6a3be0ece4 
python3 -m pip install .

#cd examples/seq2seq
#python3 -m pip install -r requirements.txt
#wget https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz
#tar -xzvf wmt_en_ro.tar.gz