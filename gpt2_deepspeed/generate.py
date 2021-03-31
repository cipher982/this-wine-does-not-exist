import argparse
import logging
import os
from pathlib import Path
import pickle

import deepspeed
import pandas as pd
import torch
from tqdm import tqdm
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
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./",
        help='Directory to save the generated text output'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
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
    model = torch.load(model_path).to('cuda:0')
    LOG.info("Loaded gpt2 model")

    # Set to eval mode?
    model = model.eval()

    # Initialize DeepSpeed
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=None,
        model_parameters=parameters
    )
    _, client_sd = model_engine.load_checkpoint(args.model, tag='step_14000')

    model_engine = model_engine.eval()
    model = model.eval()

    # Generate some samples!
    with torch.no_grad():
        generated_descriptions = []
        for ix, sample in tqdm(enumerate(wine_names)):
            prompt = f"<|startoftext|> [prompt] " + sample + " [response] "
            encoded_sample = tokenizer.encode(
                text=prompt,
                return_tensors='pt'
            ).to('cuda:0')

            try:
                output_ids = model.generate(
                    encoded_sample.to('cuda:0'),
                    do_sample=True,
                    max_length=300,
                    min_length=100,
                    top_p=0.8,
                    top_k=200,
                    temperature=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    early_stopping=True
                )

                output_text = tokenizer.decode(
                    token_ids=output_ids[0],
                    skip_special_tokens=True
                )
                LOG.info(output_text)
                err = 'None'
            except Exception as e:
                output_text='crashed'
                err = str(e)

            generated_descriptions.append((sample, output_ids, output_text, err))

            if ix % 50 == 0:
                generated_df = pd.DataFrame(
                    data=generated_descriptions,
                    columns=["name", "tokens", "output", "error_msg"]
                )

                output_path = Path(
                    args.output_dir, 
                    f"generated_desc_{len(generated_df)}.csv"
                )
                generated_df.to_csv(output_path)


if __name__ == "__main__":
	main()