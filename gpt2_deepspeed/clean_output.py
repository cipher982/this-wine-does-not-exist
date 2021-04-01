import argparse
import logging
from pathlib import Path
import sys

import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)


def add_argument():
    parser = argparse.ArgumentParser(description='clean-gpt-output')

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

    args = parser.parse_args()
    return args


def clean_dataset(data: pd.DataFrame):
    input_shape = data.shape
    data = data[data['error_msg'] == 'None']
    LOG.info(f"Removed error samples, shape:{input_shape}->{data.shape}")
    data = data[['name', 'output']].replace('"', '', regex=True)
    LOG.info("Removed superfluous double quotes")

    category_1 = data['output']\
        .str.split('\[category_1\]', expand=True)[1]\
        .str.split('\[category_2\]', expand=True)[0]

    category_2 = data['output']\
        .str.split('\[category_2]', expand=True)[1]\
        .str.split('\[origin]', expand=True)[0]

    origin = data['output']\
        .str.split('\[origin]', expand=True)[1]\
        .str.split(' from ', expand=True)[1]\
        .str.split('\[description\]', expand=True)[0]

    description = data['output']\
        .str.split('\[description]', expand=True)[1]

    data_clean = pd.DataFrame({
        'name': data['name'],
        'category_1': category_1,
        'category_2': category_2,
        'origin': origin,
        'description': description
    })
    LOG.info(f"Re-formatted dataset with shape: {data_clean.shape}")

    category_1_valids = ['Red Wine', 'White Wine',
                         'Pink and Rosé', 'Sparkling & Champagne']
    category_2_valids = ['Chardonnay', 'Pinot Noir', 'Cabernet Sauvignon',
                         'Bordeaux Red Blends', 'Other Red Blends', 'Syrah/Shiraz',
                         'Sauvignon Blanc', 'Merlot', 'Sangiovese', 'Rhone Red Blends',
                         'Zinfandel', 'Riesling', 'Rosé', 'Pinot Gris/Grigio',
                         'Other White Blends', 'Tempranillo', 'Nebbiolo', 'Malbec']

    data_clean = data_clean.applymap(
        lambda x: x.strip() if isinstance(x, str) else x)
    data_clean = data_clean[data_clean['category_1'].isin(category_1_valids)]
    data_clean = data_clean[data_clean['category_2'].isin(category_2_valids)]
    LOG.info(f"Removed non-valid categories, new shape: {data_clean.shape}")

    return data_clean


def main():
    args = add_argument()
    data_input = pd.read_csv(args.dataset)
    LOG.info(f"Loaded {len(data_input)} generated description samples")

    data_output = clean_dataset(data_input)
    LOG.info(f"Cleaned dataset with shape: {data_output.shape}")

    save_path = Path(args.output_dir, f"cleaned_gpt_descriptions_{len(data_output)}.csv")
    data_output.to_csv(save_path)
    LOG.info(f"Saved cleaned dataset to {save_path}")

if __name__ == '__main__':
    main()
