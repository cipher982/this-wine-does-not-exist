"""Data preprocessing pipeline responsible for modernizing the legacy corpus."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from .loaders import DATA_ROOT, RAW_DATASET, PROCESSED_DATASET, IMAGE_ROOT

LEGACY_PICKLE = DATA_ROOT / "raw" / "legacy" / "wine_scraped_125k.pickle.gz"
LEGACY_DESCRIPTIONS = DATA_ROOT / "raw" / "legacy" / "wine_descriptions_230k.csv.gz"
DESCRIPTIONS_PARQUET = DATA_ROOT / "raw" / "wine_descriptions_230k.parquet"
SPLITS_PATH = DATA_ROOT / "processed" / "train_val_test_splits.json"
QUALITY_REPORT_PATH = DATA_ROOT / "processed" / "data_quality_report.json"
MANIFEST_PATH = DATA_ROOT / "processed" / "wine_images_organized" / "validation" / "validation_manifest.csv"


def _normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={c: c.strip() for c in frame.columns})
    if "price" in frame.columns:
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    return frame


def migrate_legacy_sources(*, overwrite: bool = False) -> tuple[Path, Path]:
    """Convert legacy pickle/CSV sources into Parquet files."""

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DATASET.parent.mkdir(parents=True, exist_ok=True)

    if not LEGACY_PICKLE.exists():
        raise FileNotFoundError(f"Legacy pickle missing at {LEGACY_PICKLE}")

    if overwrite or not RAW_DATASET.exists():
        with pd.io.common.get_handle(LEGACY_PICKLE, "rb") as handle:
            frame = pd.read_pickle(handle.handle)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        frame = _normalize_dataframe(frame)
        frame.to_parquet(RAW_DATASET, compression="snappy", engine="pyarrow")

    if LEGACY_DESCRIPTIONS.exists() and (overwrite or not DESCRIPTIONS_PARQUET.exists()):
        desc_df = pd.read_csv(
            LEGACY_DESCRIPTIONS,
            sep="|",
            names=["url", "description"],
            compression="gzip",
            dtype=str,
            encoding="latin1",
        )
        desc_df["description"] = (
            desc_df["description"].fillna("").apply(lambda x: x.encode("latin1").decode("utf-8", "ignore"))
        )
        desc_df.to_parquet(DESCRIPTIONS_PARQUET, compression="snappy", engine="pyarrow")

    return RAW_DATASET, DESCRIPTIONS_PARQUET


TYPE_KEYWORDS = {
    "red_wine": [
        "cabernet",
        "pinot noir",
        "merlot",
        "syrah",
        "shiraz",
        "red wine",
        "malbec",
        "zinfandel",
        "bordeaux",
        "rioja",
        "chianti",
        "tempranillo",
        "sangiovese",
        "nebbiolo",
        "garnacha",
        "barolo",
        "beaujolais",
    ],
    "white_wine": [
        "chardonnay",
        "sauvignon blanc",
        "white wine",
        "pinot gris",
        "pinot grigio",
        "riesling",
        "viognier",
        "chenin blanc",
        "albariño",
    ],
    "sparkling": ["sparkling", "champagne", "cava", "prosecco", "franciacorta", "brut"],
    "dessert": ["dessert", "port", "late harvest", "sauternes", "icewine", "vin santo"],
    "rosé": ["rosé", "rose"],
}

REGION_KEYWORDS = {
    "france": ["france", "bordeaux", "burgundy", "champagne", "rhône", "loire", "alsace", "languedoc"],
    "italy": ["italy", "tuscany", "piemonte", "sicily", "veneto", "barolo", "chianti", "prosecco"],
    "california": ["california", "napa", "sonoma", "paso robles", "central coast", "russian river"],
    "spain": ["spain", "rioja", "ribera del duero", "cava", "priorat"],
    "australia": ["australia", "barossa", "mclaren vale", "margaret river"],
    "chile": ["chile", "colchagua", "maipo"],
    "argentina": ["argentina", "mendoza"],
}

PRIMARY_REGIONS = {"france", "italy", "california"}


def _match_keyword(text: str, keyword_map: dict[str, Iterable[str]], default: str) -> str:
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in text:
                return label
    return default


def build_training_dataset(*, overwrite: bool = False) -> Path:
    """Generate the processed training dataset, splits, and validation assets."""

    if not RAW_DATASET.exists():
        migrate_legacy_sources(overwrite=False)

    PROCESSED_DATASET.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(RAW_DATASET)
    frame["price"] = pd.to_numeric(frame.get("price"), errors="coerce")
    frame["image_filename"] = frame["image_path"].astype(str).apply(lambda p: Path(p).name)
    frame["name_lower"] = frame["name"].astype(str).str.lower()
    frame["desc_lower"] = frame["description"].astype(str).str.lower()
    frame["url_lower"] = frame["url"].astype(str).str.lower()

    frame["wine_category"] = [_match_keyword(f"{n} {d}", TYPE_KEYWORDS, "other") for n, d in zip(frame["name_lower"], frame["desc_lower"])]

    region_match = []
    for n, d, u in zip(frame["name_lower"], frame["desc_lower"], frame["url_lower"]):
        text = f"{n} {d} {u}"
        region = _match_keyword(text, REGION_KEYWORDS, "other")
        if region not in PRIMARY_REGIONS:
            region = "other"
        region_match.append(region)
    frame["region"] = region_match

    frame["has_image_file"] = frame["image_filename"].apply(lambda fn: (IMAGE_ROOT / fn).exists())
    frame["has_description"] = frame["description"].astype(str).str.len() > 0

    processed = frame[[
        "name",
        "description",
        "price",
        "wine_category",
        "region",
        "image_filename",
        "url",
    ]].copy()
    processed.to_parquet(PROCESSED_DATASET, compression="snappy", engine="pyarrow", index=False)

    if overwrite or not SPLITS_PATH.exists():
        shuffled = processed.sample(frac=1.0, random_state=2025)
        train_cut = int(len(shuffled) * 0.8)
        val_cut = train_cut + int(len(shuffled) * 0.1)
        indices = {
            "train": shuffled.index[:train_cut].tolist(),
            "validation": shuffled.index[train_cut:val_cut].tolist(),
            "test": shuffled.index[val_cut:].tolist(),
        }
        SPLITS_PATH.write_text(json.dumps(indices, indent=2), encoding="utf-8")

    report = {
        "total_records": int(len(frame)),
        "records_with_descriptions": int(frame["has_description"].sum()),
        "missing_descriptions": int(len(frame) - frame["has_description"].sum()),
        "records_with_image_files": int(frame["has_image_file"].sum()),
        "missing_image_files": int((~frame["has_image_file"]).sum()),
        "price": {
            "min": float(frame["price"].min(skipna=True)),
            "max": float(frame["price"].max(skipna=True)),
            "mean": float(frame["price"].mean(skipna=True)),
            "median": float(frame["price"].median(skipna=True)),
        },
        "wine_category_distribution": Counter(frame["wine_category"]).most_common(),
        "region_distribution": Counter(frame["region"]).most_common(),
    }
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    validation_dir = IMAGE_ROOT.parent / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for child in ["valid_images", "corrupted", "missing_metadata"]:
        (validation_dir / child).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in processed.itertuples(index=True):
        status = "valid" if (IMAGE_ROOT / row.image_filename).exists() else "missing_image"
        manifest_rows.append({
            "index": row.Index,
            "name": row.name,
            "wine_category": row.wine_category,
            "region": row.region,
            "image_filename": row.image_filename,
            "status": status,
        })
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)

    # Note: Removed symlink generation to keep dataset simple
    return PROCESSED_DATASET


def _refresh_symlinks(processed: pd.DataFrame) -> None:
    category_dir = IMAGE_ROOT.parent / "by_category"
    region_dir = IMAGE_ROOT.parent / "by_region"

    for directory in [category_dir, region_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if child.is_dir():
                for symlink in child.iterdir():
                    symlink.unlink()

    def ensure_link(source: Path, destination: Path) -> None:
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(source, destination.parent)
        os.symlink(rel, destination)

    for row in processed.itertuples(index=False):
        source = IMAGE_ROOT / row.image_filename
        if not source.exists():
            continue
        ensure_link(source, category_dir / row.wine_category / row.image_filename)
        ensure_link(source, region_dir / row.region / row.image_filename)


__all__ = ["migrate_legacy_sources", "build_training_dataset"]
