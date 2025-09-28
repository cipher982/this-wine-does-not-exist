#!/bin/sh

python3 deep_gpt2.py --deepspeed_config ds_config_1gpu.json  \
--save_dir /datadrive/runs \
--train_batch_size 5 \
--save_interval 2000 \
--epochs 5 
#--load_dir "/datadrive/runs" --ckpt_id "step_8750" 