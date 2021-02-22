import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#import deepspeed
#import mpi4py
#import pandas
import torch
import transformers
import wandb

#%env WANDB_PROJECT=wine_gpt2_Trainer_42

MODEL_NAME = 'gpt2-medium'


#wandb.login(anonymous='never', key="222a37baaf0c1b0d1499ec003e5c2fe49f97b107")
wandb.init()
#wandb.watch(log='all')

print(torch.cuda.is_available())
print(f"transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")


tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
print(len(tokenizer))

tokenizer.add_special_tokens(
  {'eos_token':'<|startoftext|>',
   'bos_token':'<|startoftext|>'
  }
)
tokenizer.add_tokens(['[prompt]','[response]','[category_1]',
                      '[category_2]','[origin]','[description]',
                      '<|endoftext|>'])

tokenizer.pad_token = tokenizer.eos_token

tokenizer.save_pretrained("data/modeling/trainer_42/")

print(len(tokenizer))
print("Created tokenizer")
