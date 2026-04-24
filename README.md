# Understanding Scanned Invoices Using LayoutLM with Cross-Domain Transfer

**DTSC302 Course Project — FLAME University**
Asmita Chhabra & Nandika Aggarwal

---

## Project Overview

This project builds a two-stage pipeline for automated key-field extraction from scanned invoices using **LayoutLM** — a transformer model that jointly encodes text and 2D spatial (bounding-box) features. The system is trained on a clean dataset of 500 structured invoices (D1), then transferred via fine-tuning to 80 real-world receipts and bills (D2 — "in the wild").

Structural similarity within each dataset is independently measured using **Jaccard** and **Cosine** similarity on bounding-box representations.

---

## Results Summary

| Stage | Model | Best Val Macro-F1 |
|-------|-------|-------------------|
| D1 Invoice Training | `layoutlm.py` | **0.9591** |
| D2 Wild Baseline | `train_wild.py` | 0.8270 (weighted) |
| D2 Wild Fixed | `train_wild_fix_total.py` | **0.8970** |

---

## Project Structure

```
ML2/
├── Invoice_dataset_D1/               # Dataset 1 — 500 structured invoices
│   ├── D1_raw/batch_1/batch1_1/      # Raw invoice JPG images
│   ├── D1_raw/batch_1/batch1_1.csv   # Structured JSON data per invoice
│   ├── annotations/batch1_1/
│   │   ├── annotations.json          # Bounding box annotations (output of annotate_D1.py)
│   │   └── visualizations/           # Annotated invoice images with drawn boxes
│   └── preprocessed/batch1_1/
│       └── dataset.json              # LayoutLM-ready token/bbox/label records
│
├── Into_the_wild_D2/                 # Dataset 2 — 80 real-world receipts
│   ├── 79files/                      # Raw bill JPGs + JSONs (bill_001 to bill_079)
│   ├── dataset_wild_unmapped.json    # Raw annotations with original label schema
│   └── dataset_mapped.json           # Remapped annotations using shared LABEL_MAP
│
├── models/
│   ├── layoutlm_invoices/            # Best D1-trained model checkpoint
│   ├── layoutlm_wild_v1/             # D2 baseline model checkpoint
│   └── layoutlm_wild_final/          # D2 fixed model checkpoint (best overall)
│
├── similarity_results_1/
│   ├── reults_for_invioicesimilarity.json    # D1 within-dataset similarity results
│   └── similarity_results_wild_!.json        # D2 within-dataset similarity results
│
├── annotate_D1.py                    # Step 1 — Annotate D1 invoices with bounding boxes
├── preprocss_D1.py                   # Step 2 — Preprocess D1 into LayoutLM format
├── preprocess_D2.py                  # Step 2b — Remap D2 labels to shared schema
├── layoutlm.py                       # Step 3 — Train LayoutLM on D1
├── train_wild.py                     # Step 4a — Transfer learning on D2 (baseline)
├── train_wild_fix_total.py           # Step 4b — Transfer learning on D2 (fixed, class-weighted)
├── similarity.py                     # Step 5 — Similarity detection (Cosine + Jaccard)
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

Reads each invoice image and its corresponding CSV/JSON data. Uses Tesseract OCR to locate each field on the image and draws bounding boxes around: `invoice_number`, `invoice_date`, `due_date`, `client_name`, `client_address`, `seller_name`, `seller_address`, `tax`, `total`.

```bash
python3 Invoice_dataset_D1/annotate_D1.py
```

**Saves:**
- `Invoice_dataset_D1/annotations/batch1_1/annotations.json` — bounding box annotations for all invoices
- `Invoice_dataset_D1/annotations/batch1_1/visualizations/` — one annotated image per invoice with colored boxes drawn over each detected field

---

### Step 2a — Preprocessing D1 (`preprocss_D1.py`)

Converts annotated invoices into LayoutLM-compatible records. Each word gets its OCR bounding box normalized to [0, 1000] and assigned a label integer. Resizes all images to a standard 1000×1400.

```bash
python3 Invoice_dataset_D1/preprocss_D1.py
```

**Saves:**
- `Invoice_dataset_D1/preprocessed/batch1_1/dataset.json` — final LayoutLM-ready JSON with one record per invoice, each containing `words`, `bboxes` (normalized to 0–1000), and `labels` (integer IDs)

---

### Step 2b — Label Remapping D2 (`preprocess_D2.py`)

Reads the raw wild dataset annotations and maps the original label schema (`B-vendor`, `B-date`, `B-total`, etc.) to the shared `LABEL_MAP` used by D1. This allows the D1-trained model to be directly reused for D2 fine-tuning without re-initialising the classification head.

```bash
python3 Into_the_wild_D2/preprocess_D2.py
```

**Input:** `Into_the_wild_D2/dataset_wild_unmapped.json`

**Saves:**
- `Into_the_wild_D2/dataset_mapped.json` — remapped dataset with integer label IDs matching the shared `LABEL_MAP`

---

### Step 3 — Train on D1 (`layoutlm.py`)

Fine-tunes `microsoft/layoutlm-base-uncased` on the 500 annotated invoices. Only the classifier head and top 2 encoder layers (10 & 11) are trainable. Includes bounding-box jitter, OCR noise augmentation, class-weighted loss, and label smoothing.

```bash
python3 layoutlm.py
```

**Key config:**
- Learning rate: `3e-5`
- Epochs: `2` (early stopping, patience=1)
- Batch size: `2`, Max sequence length: `128`
- Dropout: `0.3` (hidden + attention)

**Saves:**
- `models/layoutlm_D1_final/` — best D1 model checkpoint (weights + tokenizer)

---

### Step 4a — Transfer Learning D2 Baseline (`train_wild.py`)

Loads the D1 checkpoint from `models/layoutlm_D1_final/` and fine-tunes on the mapped wild dataset with standard cross-entropy loss. Achieved weighted F1 of 0.8270 but `tax` and `total` classes had F1 = 0.000 due to class imbalance.

```bash
python3 train_wild.py
```

**Saves:**
- `models/layoutlm_wild_v1/` — baseline D2 model checkpoint (weights + tokenizer)

---

### Step 4b — Transfer Learning D2 Fixed (`train_wild_fix_total.py`)

Identical to Step 4a but adds class-weighted cross-entropy loss with log-inverse-frequency weights. Resolves the `tax`/`total` collapse and achieves macro-F1 of **0.8970**. Also includes a rule-based fallback for total extraction.

```bash
python3 train_wild_fix_total.py
```

**Key config:**
- Learning rate: `1e-5`
- Epochs: `3`
- Batch size: `2`, Max sequence length: `256`
- Loss: Class-weighted CrossEntropy + label smoothing `0.1`

**Saves:**
- `models/layoutlm_wild_final/` — best D2 model checkpoint (weights + tokenizer)

---

### Step 5 — Similarity Detection (`similarity.py`)

Extracts feature vectors (CLS token embeddings via mean pooling) from all invoices using the trained LayoutLM model. Computes both Cosine and Jaccard similarity between all invoice pairs. Returns top-5 most similar invoices per query.

```bash
python3 similarity.py
```

**Config (edit at top of file):**
```python
TRAINED_MODEL_PATH   = "models/layoutlm_wild_final"             # or layoutlm_invoices for D1
DATASET_JSON         = "Into_the_wild_D2/dataset_mapped.json"   # or D1 dataset
OUTPUT_DIR           = "similarity_results_1"
SIMILARITY_THRESHOLD = 0.80    # cosine
```

**Saves:**
- `similarity_results_1/similarity_results.json` — pairwise Cosine + Jaccard scores and top-5 most similar invoices per query

---

## One-Command Pipeline (Optional)

```bash
python3 annotate_D1.py && \
python3 preprocss_D1.py && \
python3 preprocess_D2.py && \
python3 layoutlm.py && \
python3 train_wild.py && \
python3 train_wild_fix_total.py && \
python3 similarity.py
```

---

## Label Map

```python
LABEL_MAP = {
    "other":          0,
    "invoice_number": 1,
    "invoice_date":   2,
    "due_date":       3,
    "client_name":    4,
    "client_address": 5,
    "seller_name":    6,
    "seller_address": 7,
    "tax":            8,
    "total":          9,
}
```

---

## Hardware

All experiments were run on **macOS with Apple Silicon (MPS backend)**. The code automatically detects and uses MPS if available, falling back to CPU. Batch size is kept at 2 due to memory constraints.

---

## Key Findings

- D1 (homogeneous invoices) achieved macro-F1 of **0.9591** — high layout consistency aids learning
- D2 baseline transfer suffered complete failure on `tax` and `total` due to class imbalance
- Class-weighted loss in the fixed model resolved minority class collapse, reaching **0.8970**
- Cosine similarity within D1 ≈ 0.989 (highly homogeneous); within D2 ≈ 0.974 (moderately diverse)
- Jaccard scores are low in both datasets (below 0.3 threshold), indicating absolute bbox overlap is limited even among structurally similar documents

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

The trained model checkpoints are not included in this repo due to GitHub file size limits. To run the project:

1. Run `layoutlm.py` to train the D1 model — it will download `microsoft/layoutlm-base-uncased` automatically
2. Run `train_wild_fix_total.py` to fine-tune on D2
3. The raw invoice images (`D1_raw`) and wild bill images (`79files`) are also excluded due to size limits — contact the authors for the full dataset
