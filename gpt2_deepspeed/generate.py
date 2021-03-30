import argparse
import logging
import os
from pathlib import Path
import pickle

import deepspeed
import pandas as pd
import torch
import transformers

os.environ['TOKENIZERS_PARALLELISM'] = "false"

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def add_argument():
    parser = argparse.ArgumentParser(description='gpt2-wine')

    parser.add_argument(
        '--with_cuda',
        default=False,
        action='store_true',
        help='use CPU in case there\'s no GPU support'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='local rank passed from distributed launcher'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='path to model'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='path to tokenizer'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='path to dataset'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args
args = add_argument()

# Load wine names for prompts
wine_names = pd.read_pickle(args.dataset)
LOG.info(f"Loaded {len(wine_names):,} wine names")

# Load tokenizer
tokenizer = transformers.GPT2TokenizerFast.from_pretrained(args.tokenizer)
LOG.info(f"Loaded tokenizer with vocab size {tokenizer.vocab_size:,}")

# Load model
model_path = f"/home/drose/gpt2-xl_model"
assert os.path.exists(model_path)
LOG.info(f"Found cached model at {model_path}, loading. . .")
model = torch.load(model_path)
LOG.info("Loaded gpt2 model")

# Initialize DeepSpeed
parameters = filter(lambda p: p.requires_grad, model.parameters())
model_engine, optimizer, trainloader, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=parameters
)
_, client_sd = model_engine.load_checkpoint(args.model, tag='step_14000')

# Tokenize a prompt/name

# Generate a sample (beam search?)
pass