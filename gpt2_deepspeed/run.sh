#!/bin/sh

python3 deep_gpt2.py --deepspeed_config ds_config_1gpu.json  \
--save_dir /datadrive/runs \
--batch_size 3 \
--save_interval 1000
#--load_dir "/datadrive/runs" --ckpt_id "step_63_loss_13.9765625" 