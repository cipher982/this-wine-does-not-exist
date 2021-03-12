#!/bin/sh

export BS=1; USE_TF=0; BASE_DIR=/home/transformers/examples/seq2seq
python3 $BASE_DIR/finetune_trainer.py --model_name_or_path t5-small --output_dir output_dir \
--adam_eps 1e-06 --data_dir $BASE_DIR/wmt_en_ro --do_eval --do_train --evaluation_strategy=steps --freeze_embeds \
--label_smoothing 0.1 --learning_rate 3e-5 --logging_first_step --logging_steps 1000 --max_source_length 128 \
--max_target_length 128 --num_train_epochs 1 --overwrite_output_dir --per_device_eval_batch_size $BS \
--per_device_train_batch_size $BS --predict_with_generate --eval_steps 25000  --sortish_sampler \
--task translation_en_to_ro --test_max_target_length 128 --val_max_target_length 128 --warmup_steps 500 \
--n_train 2000 --n_val 500