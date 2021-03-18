#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["HOME"] = ''

#import deepspeed
#import pandas
import torch
import transformers
import wandb

#%env WANDB_PROJECT=wine_gpt2_Trainer_42

wandb.login(anonymous='never', key="222a37baaf0c1b0d1499ec003e5c2fe49f97b107")
wandb.init()
#wandb.watch(log='all')

print(torch.cuda.is_available())
print(f"transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")

tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2-medium')
print(tokenizer.vocab_size)

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

print(tokenizer.vocab_size)
print("Created tokenizer")

class wineDataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
            
  def __len__(self):
    return len(self.encodings['input_ids'])
    
  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = item['input_ids']
    return item

  
with open('data/scraped/name_desc_nlp_ready_train.txt', 'r', encoding='utf8') as file:
    wines_raw_train = file.read().splitlines()
with open('data/scraped/name_desc_nlp_ready_test.txt', 'r', encoding='utf8') as file:
    wines_raw_test = file.read().splitlines()
print("Loaded dataset")

#wines_raw_train, wines_raw_test = train_test_split(wines_raw,test_size=0.2)

#wine_encodings_train = tokenizer(wines_raw_train, max_length=200, truncation=True, padding=True)
wine_encodings_test = tokenizer(wines_raw_test, max_length=200, truncation=True, padding=True)
print("Encoded dataset")

#wine_dataset_train = wineDataset(wine_encodings_train)
wine_dataset_test = wineDataset(wine_encodings_test)
print("Created PyTorch DataSet")

#train_loader = torch.utils.data.DataLoader(wine_dataset_train)

model = transformers.AutoModelForCausalLM.from_pretrained('gpt2-medium')
model.to('cuda')
model.resize_token_embeddings(len(tokenizer))

print(f"model parameters: {model.num_parameters():,}")

training_args = transformers.TrainingArguments(
    output_dir="data/modeling/trainer_42/",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    deepspeed='data/ds_config.json'
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=wine_dataset_test,
)

trainer.train()
