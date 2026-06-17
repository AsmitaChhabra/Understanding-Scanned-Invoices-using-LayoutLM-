# Understanding Scanned Invoices Using LayoutLM with Cross-Domain Transfer


---

## Project Overview

This project builds a two-stage pipeline for automated key-field extraction from scanned invoices using **LayoutLM** — a transformer model that jointly encodes text and 2D spatial (bounding-box) features. The system is trained on a clean dataset of 1,414 structured invoices (D1), then transferred via fine-tuning to 79 real-world receipts and bills (D2 — "in the wild").

Structural similarity within each dataset is independently measured using **Jaccard** and **Cosine** similarity on LayoutLM-derived embeddings, with dataset-specific thresholds and dual-metric duplicate detection.

All training, evaluation, and label-mapping logic is driven by a single shared `config.py`, eliminating the label-map drift and evaluation leakage present in earlier versions of this pipeline.

---

## Results Summary

| Stage | Model | Metric | Score |
|-------|-------|--------|-------|
| D1 Invoice Training | `new_train.py` | Val Macro F1 | 0.6796 |
| D1 Invoice Training | `new_train.py` | **Test Macro F1 (honest)** | **0.6692** |
| D1 Invoice Training | `new_train.py` | Test Weighted F1 | 0.9069 |
| D2 Wild Transfer | `train_wild_fix_total.py` | Val Macro F1 | 0.4629 |
| D2 Wild Transfer | `train_wild_fix_total.py` | Val Weighted F1 | 0.8004 |

> **Note on methodology:** Earlier iterations of this pipeline reported D1 macro-F1 of 0.9591 and D2 weighted-F1 of 0.8970. Those numbers were inflated by a confidence-filtering bug in the evaluation loop (predictions below 0.6 confidence were silently dropped before scoring) and by val-set leakage into model selection (no held-out test set existed). The numbers above are honest: no confidence filtering, computed on a test set the model never saw during training or checkpoint selection.

---

## What Changed From the Original Pipeline

This version fixes five structural problems identified during a full pipeline audit:

1. **No held-out test set** — `new_train.py` now performs a stratified 70/15/15 split internally and locks the test set to `results/D1/test_split.json`, touched only once by `evaluate.py`.
2. **Confidence threshold leak in evaluation** — removed entirely from every `evaluate()` function. Confidence filtering is now exclusively a deployment-time decision (`DEPLOY_CONFIDENCE_THRESHOLD` in `config.py`), never a training-time one.
3. **D1 class imbalance** — class-weighted cross-entropy (log-inverse-frequency) ported into D1 training, plus minority-class upsampling (2x duplication for samples with >5% tokens in `tax`/`total`/address fields) and an extra 1.5x weight boost specifically for `tax` and `total`.
4. **Label map inconsistency** — a single 9-class `LABEL_MAP` (no `due_date`) now lives in `config.py` and is imported by every script. The old 10-class map (with `due_date` shifting all subsequent label IDs) has been fully retired.
5. **No centralized config** — `config.py` is the single source of truth for label maps, hyperparameters, file paths, and the deployment confidence threshold.

`layoutlm.py` and `train_wild.py` are retired and kept only for reference — they are superseded by `new_train.py` and `train_wild_fix_total.py` respectively, which fix the bugs above plus add gradient clipping, a linear warmup scheduler, per-class F1 logging every epoch, an overfitting gap warning, and a truncation audit.

---

## Project Structure

```
ML2/
├── config.py                          # Single source of truth — label map, hyperparams, paths
├── Invoice_dataset_D1/                # Dataset 1 — 1,414 structured invoices (multi-batch)
│   ├── D1_raw/                        # Raw invoice JPGs + CSVs, per batch
│   ├── annotations/                   # Bounding box annotations, per batch
│   │   └── <batch>/
│   │       ├── annotations.json       # Output of annotate_D1.py
│   │       └── visualizations/        # Annotated images with drawn boxes
│   └── preprocessed/
│       ├── <batch>/dataset.json       # LayoutLM-ready records, per batch
│       └── preprocessed_dataset.json  # All batches merged — read by new_train.py
│
├── Into_the_wild_D2/                  # Dataset 2 — 79 real-world receipts
│   ├── dataset_wild_unmapped.json     # Raw annotations, original label schema
│   └── dataset_mapped.json            # Remapped to shared 9-class LABEL_MAP
│
├── models/
│   ├── layoutlm_D1_final/             # Current D1 checkpoint (1,414 samples, clean eval)
│   └── layoutlm_wild_final/           # Current D2 checkpoint (transfer-learned from D1)
│
├── results/
│   ├── D1/
│   │   ├── test_split.json            # Locked test set — touched only by evaluate.py
│   │   ├── training_log_val_*.csv     # Per-epoch metrics
│   │   ├── summary_val_*.txt          # Best-epoch per-class breakdown
│   │   ├── eval_test_*.txt            # Final honest test-set report
│   │   └── confusion_matrix_test_*.csv
│   └── D2/
│       ├── training_log_*.csv
│       └── summary_*.txt
│
├── similarity_results_1/
│   ├── similarity_results_D1.json     # D1 similarity + likely_duplicates
│   └── similarity_results_D2.json     # D2 similarity + likely_duplicates
│
├── annotate_D1.py                     # Step 1 — Annotate D1 invoices with bounding boxes
├── preprocss_D1.py                    # Step 2 — Preprocess D1 into LayoutLM format
├── preprocess_D2.py                   # Step 2b — Remap D2 labels to shared 9-class schema
├── new_train.py                       # Step 3 — Train LayoutLM on D1 (current, active)
├── train_wild_fix_total.py            # Step 4 — Transfer learning on D2 (current, active)
├── evaluate.py                        # Step 5 — Final honest test-set evaluation (run once)
├── similarity.py                      # Step 6 — Similarity + duplicate detection
├── layoutlm.py                        # RETIRED — superseded by new_train.py
├── train_wild.py                      # RETIRED — superseded by train_wild_fix_total.py
└── requirements.txt
```

---

## Installation

### 1. Install Tesseract

**macOS**
```bash
brew install tesseract
```

> If you don't have Homebrew:
> ```bash
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> ```

**Windows**

Download and run the installer from:
https://github.com/UB-Mannheim/tesseract/wiki

During installation, note the path (usually `C:\Program Files\Tesseract-OCR`). Then add it to your system PATH:
1. Search "Environment Variables" in the Start menu
2. Under System Variables → find `Path` → click Edit
3. Add `C:\Program Files\Tesseract-OCR`
4. Click OK and restart your terminal

---

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

### 3. Create and Activate Virtual Environment

**macOS / Linux**
```bash
python3 -m venv invoice_env
source invoice_env/bin/activate
```

**Windows**
```bash
python -m venv invoice_env
invoice_env\Scripts\activate
```

---

### 4. Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Pipeline — Step by Step

### Step 1 — Annotation (`annotate_D1.py`)

Reads each invoice image and its corresponding CSV/JSON data. Uses Tesseract OCR to locate each field on the image and draws bounding boxes around: `invoice_number`, `invoice_date`, `client_name`, `client_address`, `seller_name`, `seller_address`, `tax`, `total`.

> `due_date` is intentionally not annotated or trained on — it was present in earlier label maps but never had real coverage in the data, and was the source of a 10-class vs 9-class label ID mismatch across scripts. It has been fully removed from the shared label map.

```bash
python3 annotate_D1.py
```

**Saves:**
- `Invoice_dataset_D1/annotations/<batch>/annotations.json` — bounding box annotations for all invoices in that batch
- `Invoice_dataset_D1/annotations/<batch>/visualizations/` — one annotated image per invoice with colored boxes drawn over each detected field

---

### Step 2a — Preprocessing D1 (`preprocss_D1.py`)

Converts annotated invoices into LayoutLM-compatible records. Each word gets its OCR bounding box normalized to [0, 1000] and assigned a label integer from the shared `LABEL_MAP`. Resizes all images to a standard 1000×1400.

```bash
python3 preprocss_D1.py
```

**Saves:**
- `Invoice_dataset_D1/preprocessed/<batch>/dataset.json` — one record per invoice, each containing `words`, `bboxes` (normalized to 0–1000), and `labels` (integer IDs, 9-class)

To add a new batch later, run `annotate_D1.py` and `preprocss_D1.py` against the new batch's raw folder. `new_train.py` auto-discovers every `dataset.json` under `preprocessed/` via `glob` — no merge step or config edit is required.

---

### Step 2b — Label Remapping D2 (`preprocess_D2.py`)

Reads the raw wild dataset annotations and maps the original 11-class label schema (`B-vendor`, `B-date`, `B-total`, `B-cgst`, `B-sgst`, `B-gstin`, etc.) to the shared 9-class `LABEL_MAP` imported from `config.py`. `B-cgst`, `B-sgst`, and `B-gstin` all collapse into the single `tax` class, since D1 only has one tax label. This allows the D1-trained classification head to be reused directly for D2 fine-tuning.

```bash
python3 preprocess_D2.py
```

**Input:** `Into_the_wild_D2/dataset_wild_unmapped.json`

**Saves:**
- `Into_the_wild_D2/dataset_mapped.json` — remapped dataset with integer label IDs matching the shared 9-class `LABEL_MAP`

---

### Step 3 — Train on D1 (`new_train.py`)

Fine-tunes `microsoft/layoutlm-base-uncased` on 1,414 annotated invoices (auto-aggregated across all batches). Encoder layers 8–11 plus the classifier head are trainable; layers 0–7 stay frozen. Includes bounding-box jitter, OCR noise augmentation, class-weighted loss with an extra boost for `tax`/`total`, minority-class sample upsampling, label smoothing, gradient clipping, and a linear warmup scheduler.

```bash
python3 new_train.py
```

**Key config (from `config.py`):**
- Learning rate: `2e-5`
- Epochs: `15` (early stopping, patience=5)
- Batch size: `2`, Max sequence length: `512`
- Dropout: `0.1` (hidden + attention)
- Label smoothing: `0.05`

**Saves:**
- `models/layoutlm_D1_final/` — best D1 model checkpoint by validation macro-F1 (weights + tokenizer)
- `results/D1/test_split.json` — the locked held-out test set, written once at the start of training and never touched again until `evaluate.py` runs

**Per-class test F1 (final, honest):**

| Field | Precision | Recall | F1 |
|---|---|---|---|
| other | 0.973 | 0.963 | 0.968 |
| invoice_number | 1.000 | 1.000 | 1.000 |
| invoice_date | 0.995 | 1.000 | 0.998 |
| client_name | 0.611 | 0.655 | 0.632 |
| client_address | 0.521 | 0.372 | 0.434 |
| seller_name | 0.654 | 0.731 | 0.690 |
| seller_address | 0.471 | 0.401 | 0.433 |
| tax | 0.430 | 0.669 | 0.523 |
| total | 0.280 | 0.449 | 0.345 |

`invoice_number` and `invoice_date` are solved fields. `client_address`/`seller_address` are weakened by spatial confusion with the corresponding `_name` field (the model's top error mode — 823 and 683 confusions respectively, per the confusion matrix). `total` is the weakest field by support (401 test tokens) and is partially compensated downstream by a rule-based fallback in the D2 transfer script.

---

### Step 4 — Transfer Learning D2 (`train_wild_fix_total.py`)

Loads the D1 checkpoint from `models/layoutlm_D1_final/` and fine-tunes on the 79-sample mapped wild dataset. Adds class-weighted cross-entropy loss with log-inverse-frequency weights, resolving the complete `tax`/`total` collapse seen under standard cross-entropy. Only encoder layer 11 plus the classifier is unfrozen — deliberately more conservative than D1's 4 unfrozen layers, since 79 samples risks catastrophic forgetting of what D1 already learned if too much capacity is unfrozen. Also includes a rule-based fallback for total extraction (keyword search for `TOTAL`/`GRAND`/`PAYABLE`/`AMOUNT` near numeric tokens) that runs independently of the model's per-token predictions.

```bash
python3 train_wild_fix_total.py
```

**Key config (from `config.py`):**
- Learning rate: `1e-5`
- Epochs: `5` (early stopping, patience=3)
- Batch size: `2`, Max sequence length: `512`
- Loss: Class-weighted CrossEntropy + label smoothing `0.05`

**Saves:**
- `models/layoutlm_wild_final/` — best D2 model checkpoint (weights + tokenizer)
- `results/D2/training_log_*.csv`, `results/D2/summary_*.txt`

> 79 samples is a hard ceiling for this stage. The model was still improving at the final epoch with no sign of early stopping, but per-class F1 on `tax` (0.282) and `total` (0.103) reflects genuine data scarcity rather than a methodology problem. The rule-based total extraction fallback already produces sane outputs (verified manually against 5 validation samples) independent of model F1 on the `total` label. This will improve naturally once more wild receipts are collected and added.

---

### Step 5 — Final Evaluation (`evaluate.py`)

Runs exactly once, on the locked test set written by `new_train.py`. No confidence filtering. Produces a classification report, confusion matrix, top-10 misclassification pairs, and a confidence distribution across thresholds (0.5–0.95) to inform the deployment cutoff.

```bash
python3 evaluate.py
```

**Saves:**
- `results/D1/eval_test_*.txt` — final honest classification report
- `results/D1/confusion_matrix_test_*.csv`

**Confidence distribution (test set, 58,640 predictions):**

| Threshold | % of predictions above |
|---|---|
| ≥ 0.50 | 99.0% |
| ≥ 0.60 | 96.3% |
| ≥ 0.70 | 93.5% |
| ≥ 0.80 | 39.6% |
| ≥ 0.90 | 8.4% |

The sharp drop between 0.70 and 0.80 suggests **0.70** as a reasonable deployment confidence threshold (`DEPLOY_CONFIDENCE_THRESHOLD` in `config.py`) — predictions below it should be flagged for human review rather than trusted outright.

---

### Step 6 — Similarity & Duplicate Detection (`similarity.py`)

Uses the trained LayoutLM (base model, no classification head) as a feature extractor — mean-pools the last hidden state across all tokens to get one embedding per invoice. Computes both Cosine similarity (layout + semantic) and Jaccard similarity (raw word overlap) between every invoice pair, per dataset.

```bash
python3 similarity.py
```

**Per-dataset cosine thresholds** (configured in `similarity.py`, not `config.py` since these are similarity-specific tuning knobs):

| Dataset | Cosine threshold | Why |
|---|---|---|
| D1_invoices | 0.995 | D1 invoices share a near-identical template; embeddings cluster tightly (observed mean cosine ≈ 0.978). A 0.80 threshold flags nearly every pair as "similar," which is not a useful signal. |
| D2_wild | 0.80 | D2 receipts vary genuinely by vendor and layout (observed mean cosine ≈ 0.885), so 0.80 is meaningful here. |

A pair is only flagged as a **likely duplicate** when both cosine and Jaccard agree above their respective thresholds — a much stronger signal than either metric alone, since high cosine alone can just mean "same template" rather than "same document."

**Saves (per dataset):**
- `similarity_results_1/similarity_results_D1.json`
- `similarity_results_1/similarity_results_D2.json`

Each file contains `threshold_used`, a `likely_duplicates` list, and the full top-5 per-query results for both metrics.

**Findings from the current run:**
- D1: 1 likely duplicate found — `batch1-0575.jpg` matched against itself at cosine 1.0 / Jaccard 1.0, suggesting this file may be present twice in the raw image folder. Worth a manual check before the next training run.
- D2: 1 likely duplicate found — `bill_045` ↔ `bill_052` at cosine 0.9938 / Jaccard 0.9242, suggesting either a re-scanned duplicate or two receipts from the same vendor with near-identical line items.

---

## Configuration (`config.py`)

Single source of truth for the entire pipeline. Every script imports from here — no script defines its own copy of the label map, hyperparameters, or file paths.

```python
LABEL_MAP = {
    "other":          0,
    "invoice_number": 1,
    "invoice_date":   2,
    "client_name":    3,
    "client_address": 4,
    "seller_name":    5,
    "seller_address": 6,
    "tax":            7,
    "total":          8,
}
```

Also defines: `MAX_SEQ_LEN`, `BATCH_SIZE`, `VAL_SPLIT`/`TEST_SPLIT`, D1 and D2 learning rates/epochs/patience, `DEPLOY_CONFIDENCE_THRESHOLD`, and all model/results/data directory paths.

---

## Hardware

All experiments were run on **macOS with Apple Silicon (M1, 8GB RAM, MPS backend)**. The code automatically detects and uses MPS if available, falling back to CUDA, then CPU. Batch size is kept at 2 and only 1–4 encoder layers are ever unfrozen at once to keep training within memory constraints on consumer hardware.

---

## Key Findings

- D1's near-identical invoice template makes layout-based fields (`invoice_number`, `invoice_date`) effectively solved (F1 ≥ 0.998), while spatially-adjacent fields (`client_name`/`client_address`, `seller_name`/`seller_address`) remain the dominant error mode — the model confuses names and addresses far more often than it confuses unrelated field types.
- Class-weighted loss alone was insufficient for `tax`/`total` recovery; combining it with minority-class sample upsampling and an extra weight multiplier produced a measurable (if partial) improvement (`total` F1: 0.299 → 0.345 on test).
- D2 transfer learning on 79 samples is data-bound, not method-bound — the rule-based fallback for `total` extraction exists specifically because per-token model F1 on rare fields cannot be expected to be reliable at this sample size.
- Cosine similarity requires dataset-specific thresholds to be meaningful: a fixed 0.80 cutoff that works for diverse D2 receipts is meaningless for D1's visually homogeneous invoices, where it flags effectively the entire dataset as "similar."
- Requiring agreement between Cosine and Jaccard similarity (rather than trusting either alone) is a more reliable duplicate-detection signal, surfacing exactly one credible duplicate per dataset rather than hundreds of template-level false positives.

---

## References

- Xu et al. (2020) — LayoutLM: Pre-training of text and layout for document image understanding
- Xu et al. (2021) — LayoutLMv2: Multi-modal pre-training for visually-rich document understanding
- Huang et al. (2022) — LayoutLMv3: Pre-training for document AI with unified text and image masking
- Devlin et al. (2019) — BERT: Pre-training of deep bidirectional transformers
- Palm et al. (2017) — CloudScan: A configuration-free invoice analysis system using RNNs
- Liu et al. (2019) — Graph convolution for multimodal information extraction from visually rich documents
- Lee et al. (2020) — BioBERT: A pre-trained biomedical language representation model

---

## Note for Reviewers

Trained model checkpoints are not included in this repo due to GitHub file size limits. To reproduce:

1. Run `annotate_D1.py` then `preprocss_D1.py` for each raw batch under `Invoice_dataset_D1/D1_raw/`
2. Run `new_train.py` to train the D1 model — it will download `microsoft/layoutlm-base-uncased` automatically and write the locked test split
3. Run `preprocess_D2.py` to (re)generate the 9-class mapped D2 dataset
4. Run `train_wild_fix_total.py` to fine-tune on D2
5. Run `evaluate.py` once for the honest final D1 test metric
6. Run `similarity.py` for duplicate detection and structural similarity analysis

Raw invoice images (`D1_raw/`) and wild bill images are excluded due to size limits — contact the authors for the full dataset.
