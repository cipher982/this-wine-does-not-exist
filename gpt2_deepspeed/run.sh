#!/bin/sh

python3 deep_gpt2.py --deepspeed_config ds_config_1gpu.json  \
--save_dir /datadrive/runs \
--train_batch_size 2 \
--save_interval 500 \
--epochs 5
#--load_dir "/datadrive/runs" --ckpt_id "step_63_loss_13.9765625" 