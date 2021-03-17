import argparse
import dill
import os
import pickle
import torch
import transformers
import deepspeed

MODEL_TYPE = 'distilgpt2'

# Setup PyTorch Dataset subclass
class wineDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
            
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

# Load wine dataset
wines_path = "/content/drive/MyDrive/data/wine/name_desc_nlp_ready_test.txt"
with open(wines_path, 'r') as f:
    wines_raw = f.read().splitlines()
print(f"Loaded wine dataset of length: {len(wines_raw):,}")

# Remove wines with too short descriptions
wines_clean = []
for i in wines_raw:
    try:
        desc = i.split("[description]")[1]
        if len(desc) > 100:
            wines_clean.append(i)
    except:
        pass
print(f"Cleaned dataset has {len(wines_clean):,} samples")

tokenizer = transformers.GPT2TokenizerFast.from_pretrained(MODEL_TYPE)

tokenizer.add_special_tokens(
    {'eos_token':'<|startoftext|>',
     'bos_token':'<|startoftext|>'
    }
)
tokenizer.add_tokens(['[prompt]','[response]','[category_1]',
                      '[category_2]','[origin]','[description]',
                      '<|endoftext|>'])
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.save_pretrained('data/modeling/gpt2_distil_model/')
#tokenizer.save_pretrained('drive/MyDrive/data/wine/gpt2_large/')
print("Created tokenizer")

wine_encodings = tokenizer(wines_clean, max_length=300, padding=True, truncation=True)
#wine_encodings_train = tokenizer(wines_clean_train, max_length=300, padding=True, truncation=True)
#wine_encodings_test = tokenizer(wines_raw_test, max_length=200, padding=True, truncation=True)
print("Encoded dataset")

wine_dataset = wineDataset(wine_encodings)
#wine_dataset_train = wineDataset(wine_encodings_train)
#wine_dataset_test = wineDataset(wine_encodings_test)
print("Created PyTorch DataSet")

data_loader = torch.utils.data.DataLoader(wine_dataset, num_workers=0)
print("Created DataLoader")

# Load model
model_path = f'/root/{MODEL_TYPE}_model'
if os.path.exists(model_path):
    print(f"Found saved model at {model_path}, loading. . .")
    model = torch.load(model_path)
    print("Loaded gpt2 model")
else:
    print(f"Saved model not found, downloading. . .")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_TYPE)
    print("Loaded gpt2 model")

    # Set config
    model.config.use_cache = False
    model.config.gradient_checkpointing = True

    # Resize for tokens
    model.resize_token_embeddings(len(tokenizer))
    print("Resized token embeddings")

    # Send to GPU, set train mode, and save
    #model.to('cuda')
    #model = model.train()
    torch.save(model, model_path)
    print(f"Saved model to {model_path}")

print(f"Total parameters: {model.num_parameters()/1e6:.2f}M")

def add_argument():

    parser = argparse.ArgumentParser(description='gpt2-wine')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=1,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=1,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args
args = add_argument()

parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=parameters,
    #training_data=data_loader
)

print("YAYYY!!")
