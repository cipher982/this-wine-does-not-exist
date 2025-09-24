# Wine Dataset Classification: SOURCE vs DERIVATIVE

## 🔴 SOURCE DATA (Cannot be recreated - MUST PRESERVE)

### Primary Wine.com Scraped Data
These are the original scraped records from wine.com - the foundation of everything:

| File | Size | Content | Status |
|------|------|---------|--------|
| `data/scraped/scraped_with_decs.pickle.gz` | 62MB | 125,787 wines with full metadata | **CRITICAL SOURCE** |
| `data/scraped/scraped_with_decs.pickle.gzip` | 55MB | Duplicate of above | DUPLICATE - can delete |
| `data/scraped/scrape.pickle.gz` | 45MB | Earlier/partial scrape | **SOURCE** (older version) |
| `data/scraped/descriptions.csv.gz` | 22MB | 230,487 wine descriptions | **SOURCE** (unique descriptions) |

**Verdict**: These contain the original wine.com HTML, prices, URLs, descriptions that CANNOT be re-scraped (site has changed).

---

## 🟡 DERIVATIVE DATA (Can be recreated from source)

### Processing Derivatives
Generated from SOURCE data - can be recreated with scripts:

| File | Size | Content | Recreation Method |
|------|------|---------|-------------------|
| `data/scraped/name_desc_nlp_ready.txt` | 55MB | GPT2 training format | Process from scraped_with_decs.pickle |
| `data/scraped/name_desc_nlp_ready.txt.gz` | 18MB | Compressed version | Compress the .txt file |
| `data/cleaned_dataset.gzip` | 8.9MB | Cleaned/filtered dataset | Run cleaning notebook on source |

### Generated Fake Data
AI-generated outputs from trained models:

| File | Size | Content | Recreation Method |
|------|------|---------|-------------------|
| `data/fake/fake_names_*.pickle` | ~500KB each | Generated wine names | Re-run LSTM model |
| `data/fake/fake_names_*.csv` | Various | Generated names in CSV | Convert pickle to CSV |
| `data/fake/DESC_v*.csv` | Various | Generated descriptions | Re-run GPT2 model |
| `data/fake/generated_desc_*.csv` | Various | GPT2 outputs | Re-generate with model |
| `data/fake/cleaned_gpt_descriptions_*.csv` | Various | Filtered GPT2 outputs | Filter generated descriptions |

### Model Weights
Trained model parameters:

| File | Size | Content | Recreation Method |
|------|------|---------|-------------------|
| `data/models_weights/model_weights_name.h5` | 1.3MB | LSTM name model | Retrain from scratch |
| `data/models_weights/model_description_weights.h5` | 17MB | Description model | Retrain from scratch |
| `data/models_weights/model_char_DESCS.json` | 8.4KB | Model config | Regenerate from code |

---

## 📁 RECOMMENDED DIRECTORY STRUCTURE

```
this-wine-does-not-exist/
├── .gitignore                    # MUST ignore all data/
├── DATA_CLASSIFICATION.md        # This file
├── data/                         # ALL GITIGNORED
│   ├── 00_SOURCE/               # Original scraped data (PRESERVE)
│   │   ├── wine_scraped_125k.pickle.gz
│   │   ├── wine_descriptions_230k.csv.gz
│   │   └── README.md            # Document what each file contains
│   ├── 01_PROCESSED/            # Derivatives from source
│   │   ├── nlp_ready/
│   │   ├── cleaned/
│   │   └── README.md            # Document processing steps
│   ├── 02_GENERATED/            # Model outputs
│   │   ├── fake_names/
│   │   ├── fake_descriptions/
│   │   └── README.md            # Document generation parameters
│   └── 03_MODELS/               # Trained weights
│       ├── lstm_names/
│       ├── gpt2_descriptions/
│       └── stylegan2_labels/    # Future
├── notebooks/                    # Processing code (IN GIT)
├── src/                         # Source code (IN GIT)
└── scripts/                     # Processing scripts (IN GIT)
```

---

## 🚀 ACTION PLAN

### Phase 1: Backup Everything
```bash
# Create master backup before any changes
tar -czf ~/wine-project-backup-$(date +%Y%m%d).tar.gz .
```

### Phase 2: Reorganize Data Locally
```bash
# Create new structure
mkdir -p data/{00_SOURCE,01_PROCESSED,02_GENERATED,03_MODELS}

# Move SOURCE data
mv data/scraped/scraped_with_decs.pickle.gz data/00_SOURCE/wine_scraped_125k.pickle.gz
mv data/scraped/descriptions.csv.gz data/00_SOURCE/wine_descriptions_230k.csv.gz

# Move DERIVATIVE data
mv data/scraped/name_desc_nlp_ready.txt* data/01_PROCESSED/
mv data/cleaned_dataset.gzip data/01_PROCESSED/
mv data/fake/* data/02_GENERATED/
mv data/models_weights/* data/03_MODELS/
```

### Phase 3: Update .gitignore
```gitignore
# Add to .gitignore
data/
!data/README.md
!data/*/README.md
*.pickle
*.pickle.*
*.gzip
*.gz
*.h5
wine-dataset-*.tar.gz
```

### Phase 4: Clean Git History
```bash
# Remove ALL data files from git history (DESTRUCTIVE - backup first!)
git filter-repo --path-glob 'data/*.pickle*' --invert-paths
git filter-repo --path-glob 'data/*.gzip' --invert-paths
git filter-repo --path-glob 'data/*.gz' --invert-paths
git filter-repo --path-glob 'data/*.h5' --invert-paths
git filter-repo --path-glob '*.tar.gz' --invert-paths
```

### Phase 5: Storage Strategy
- **Local Development**: Keep full dataset in `data/` (gitignored)
- **Backup**: Copy SOURCE data to Cube/NAS
- **Archive**: Compress old derivatives, keep only SOURCE + latest generated

---

## 📊 SUMMARY

### Must Preserve (SOURCE)
- `scraped_with_decs.pickle.gz` - 125k wines with metadata
- `descriptions.csv.gz` - 230k unique descriptions

### Can Delete (DUPLICATES)
- `scraped_with_decs.pickle.gzip` - exact duplicate
- Multiple versions of same fake data in different formats

### Can Recreate (DERIVATIVES)
- All NLP-ready text files
- All generated fake names/descriptions
- All model weights (retrain with modern architecture anyway)

**Total SOURCE data to preserve: ~100MB compressed**
**Total DERIVATIVE data: ~200MB (can all be regenerated)**