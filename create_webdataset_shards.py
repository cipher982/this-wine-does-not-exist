#!/usr/bin/env python3
"""
Create WebDataset TAR shards from wine images for HuggingFace Hub upload.
Implements the recommended approach for large image datasets.
"""

import os
import tarfile
import math
import glob
import json
from pathlib import Path

def create_wine_webdataset_shards(images_dir="~/datasets/wine-images-126k/images",
                                  output_dir="~/datasets/wine-images-126k/shards"):
    """Create WebDataset TAR shards from wine images."""

    # Expand paths
    images_dir = os.path.expanduser(images_dir)
    output_dir = os.path.expanduser(output_dir)

    print("🔧 Creating WebDataset TAR shards for wine images")
    print(f"📁 Source: {images_dir}")
    print(f"🎯 Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all wine images
    pattern = os.path.join(images_dir, "wine_*.jpg")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"❌ No wine_*.jpg files found in {images_dir}")
        return False

    print(f"📊 Found {len(files):,} wine images")

    # Calculate sharding
    shard_size = 10000  # ~10k images per shard ≈ 0.5-1.0 GB at 57KB avg
    num_shards = math.ceil(len(files) / shard_size)

    print(f"📦 Creating {num_shards} shards ({shard_size:,} images per shard)")

    # Create metadata for tracking
    shard_metadata = []

    # Create shards
    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f"wine-images-{shard_idx:05d}.tar")
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(files))
        shard_files = files[start_idx:end_idx]

        print(f"📦 Shard {shard_idx+1:02d}/{num_shards}: {len(shard_files):,} images → {os.path.basename(shard_path)}")

        # Create TAR file
        with tarfile.open(shard_path, "w") as tar:
            for file_path in shard_files:
                # Use just the filename as archive name (wine_000123.jpg)
                arcname = os.path.basename(file_path)
                tar.add(file_path, arcname=arcname)

        # Calculate shard size
        shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)

        # Track shard metadata
        shard_info = {
            "shard_name": f"wine-images-{shard_idx:05d}.tar",
            "shard_path": f"shards/wine-images-{shard_idx:05d}.tar",
            "image_count": len(shard_files),
            "size_mb": round(shard_size_mb, 1),
            "start_image": os.path.basename(shard_files[0]),
            "end_image": os.path.basename(shard_files[-1])
        }
        shard_metadata.append(shard_info)

        print(f"  ✅ Created: {shard_size_mb:.1f}MB")

    # Save shard metadata
    metadata_path = os.path.join(output_dir, "shard_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            "total_images": len(files),
            "total_shards": num_shards,
            "images_per_shard": shard_size,
            "shards": shard_metadata
        }, f, indent=2)

    # Calculate total size
    total_size = sum(info["size_mb"] for info in shard_metadata)

    print(f"\n✅ WebDataset shards created successfully!")
    print(f"📊 Summary:")
    print(f"  Total images: {len(files):,}")
    print(f"  Total shards: {num_shards}")
    print(f"  Total size: {total_size:.1f}MB")
    print(f"  Average shard: {total_size/num_shards:.1f}MB")
    print(f"  Metadata: {metadata_path}")

    return True

def create_upload_script():
    """Create script to upload shards to HuggingFace Hub."""

    script_content = '''#!/bin/bash
# Upload WebDataset shards to HuggingFace Hub
# Run this script from cube server after sharding is complete

REPO_ID="cipher982/wine-images-126k"
SHARDS_DIR="~/datasets/wine-images-126k/shards"

echo "🚀 Uploading WebDataset shards to HuggingFace Hub"
echo "📦 Repository: $REPO_ID"
echo "📁 Shards directory: $SHARDS_DIR"

# Expand path
SHARDS_DIR=$(eval echo $SHARDS_DIR)

# Check if shards exist
if [ ! -d "$SHARDS_DIR" ]; then
    echo "❌ Shards directory not found: $SHARDS_DIR"
    exit 1
fi

# Count shards
SHARD_COUNT=$(find "$SHARDS_DIR" -name "wine-images-*.tar" | wc -l)
echo "📊 Found $SHARD_COUNT shard files"

if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "❌ No shard files found in $SHARDS_DIR"
    exit 1
fi

# Check HF authentication
export PATH=~/.local/bin:$PATH
if ! hf auth whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to HuggingFace"
    echo "💡 Please run: hf auth login --token YOUR_TOKEN"
    exit 1
fi

echo "✅ HuggingFace authentication verified"

# Create repository if it doesn't exist
echo "📝 Creating repository (if needed)..."
hf repo create "$REPO_ID" --type=dataset --exist-ok

# Upload shards directory
echo "📤 Uploading shards..."
cd "$SHARDS_DIR/.."
hf upload "$REPO_ID" shards --repo-type=dataset

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Upload completed successfully!"
    echo "🔗 Repository: https://huggingface.co/datasets/$REPO_ID"
    echo ""
    echo "📖 Usage example:"
    echo "import webdataset as wds"
    echo "url = 'hf://datasets/$REPO_ID/shards/wine-images-{00000..$(printf "%05d" $((SHARD_COUNT-1)))}.tar'"
    echo "ds = wds.WebDataset(url).decode('pil')"
else
    echo "❌ Upload failed with exit code: $EXIT_CODE"
fi
'''

    script_path = "upload_wine_shards.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"📤 Created upload script: {script_path}")
    return script_path

def main():
    """Main execution function."""

    print("🍷 Wine Images WebDataset Sharding")
    print("=" * 50)
    print("Following HuggingFace best practices for large image datasets")
    print()

    # This script is designed to run on cube server where images are located
    print("💡 This script should be run on cube server:")
    print("   scp create_webdataset_shards.py cube:~/")
    print("   ssh cube 'python3 ~/create_webdataset_shards.py'")
    print()

    # Create upload script for later use
    upload_script = create_upload_script()

    print(f"📋 Next steps:")
    print(f"1. Copy this script to cube: scp {__file__} cube:~/")
    print(f"2. Run sharding on cube: ssh cube 'python3 ~/{os.path.basename(__file__)}'")
    print(f"3. Copy upload script: scp {upload_script} cube:~/")
    print(f"4. Upload shards: ssh cube 'bash ~/{upload_script}'")

if __name__ == "__main__":
    # Check if we're running on cube (has the images directory)
    if os.path.exists(os.path.expanduser("~/datasets/wine-images-126k/images")):
        # We're on cube - create the shards
        create_wine_webdataset_shards()
    else:
        # We're on local machine - show instructions
        main()