#!/usr/bin/env python3
"""
Download wine bottle images from wine.com using the scraped image paths.
This script downloads all ~110k unique wine bottle images for StyleGAN2 training.
"""

import pandas as pd
import pickle
import gzip
import requests
import os
from pathlib import Path
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_image(args):
    """Download a single image with error handling."""
    idx, image_path, output_dir, session = args

    # Create filename from image path
    filename = os.path.basename(image_path)
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        filename += '.jpg'

    output_path = output_dir / filename

    # Skip if already exists
    if output_path.exists():
        return {'success': True, 'idx': idx, 'path': str(output_path), 'status': 'exists'}

    url = f'https://www.wine.com{image_path}'

    try:
        response = session.get(url, timeout=10, stream=True)
        response.raise_for_status()

        # Check if it's actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return {'success': False, 'idx': idx, 'error': f'Not an image: {content_type}', 'url': url}

        # Write image
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify file size
        if output_path.stat().st_size < 1000:  # Less than 1KB is suspicious
            output_path.unlink()  # Delete the file
            return {'success': False, 'idx': idx, 'error': 'File too small', 'url': url}

        return {'success': True, 'idx': idx, 'path': str(output_path), 'status': 'downloaded'}

    except Exception as e:
        return {'success': False, 'idx': idx, 'error': str(e), 'url': url}

def main():
    """Main download function."""
    # Load dataset
    logger.info("Loading wine dataset...")
    with gzip.open('wine_scraped_125k.pickle.gz', 'rb') as f:
        wines_df = pickle.load(f)

    # Get unique image paths
    unique_paths = wines_df['image_path'].dropna().unique()
    logger.info(f"Found {len(unique_paths)} unique image paths")

    # Create output directory
    output_dir = Path('../01_PROCESSED/wine_images')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    # Prepare download arguments
    download_args = [(i, path, output_dir, session) for i, path in enumerate(unique_paths)]

    # Download with progress bar and threading
    successful_downloads = 0
    failed_downloads = 0

    logger.info(f"Starting download of {len(download_args)} images...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all downloads
        future_to_idx = {executor.submit(download_image, args): args[0] for args in download_args}

        # Process results with progress bar
        with tqdm(total=len(download_args), desc="Downloading images") as pbar:
            for future in as_completed(future_to_idx):
                result = future.result()

                if result['success']:
                    successful_downloads += 1
                    if result.get('status') == 'downloaded':
                        pbar.set_postfix({'✓': successful_downloads, '✗': failed_downloads})
                else:
                    failed_downloads += 1
                    logger.warning(f"Failed download {result['idx']}: {result.get('error', 'Unknown error')}")
                    pbar.set_postfix({'✓': successful_downloads, '✗': failed_downloads})

                pbar.update(1)

    # Final statistics
    logger.info(f"Download complete!")
    logger.info(f"Successful: {successful_downloads}")
    logger.info(f"Failed: {failed_downloads}")
    logger.info(f"Success rate: {successful_downloads/(successful_downloads+failed_downloads)*100:.1f}%")

    # Check final directory size
    image_files = list(output_dir.glob('*.jpg')) + list(output_dir.glob('*.jpeg')) + list(output_dir.glob('*.png'))
    total_size = sum(f.stat().st_size for f in image_files) / (1024*1024*1024)  # GB

    logger.info(f"Downloaded {len(image_files)} images")
    logger.info(f"Total size: {total_size:.2f} GB")
    logger.info(f"Images saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()