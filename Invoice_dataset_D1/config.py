# =============================================================================
# config.py  —  Single source of truth for the LayoutLM invoice pipeline
# Every other script imports from here. Never hardcode these values elsewhere.
# =============================================================================

import os

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42

# ─── LABEL MAP ────────────────────────────────────────────────────────────────
# 9 classes. due_date is intentionally excluded — not annotated in D1,
# not mapped in D2. Removing it here fixes the label-shift bug that made
# evaluate.py (9 classes) incompatible with train scripts (10 classes).

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

ID2LABEL   = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)   # 9

# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
MAX_SEQ_LEN = 512             # token sequence length fed to LayoutLM
TARGET_SIZE = (1000, 1400)    # image resize (width, height) in preprocss_D1.py

# ─── SPLIT RATIOS ─────────────────────────────────────────────────────────────
VAL_SPLIT  = 0.15             # 15 % validation
TEST_SPLIT = 0.15             # 15 % test  (locked away, touched once by evaluate.py)
# train = remaining 70 %

# ─── BATCH SIZES ──────────────────────────────────────────────────────────────
BATCH_SIZE = 2

# ─── D1 TRAINING HYPERPARAMETERS ──────────────────────────────────────────────
D1_LEARNING_RATE = 2e-5       # reduced from 3e-5 — safer with 4 layers unfrozen
D1_NUM_EPOCHS    = 15         # trend was still rising at epoch 5
D1_PATIENCE      = 5          # don't stop too early

# ─── D2 TRANSFER LEARNING HYPERPARAMETERS ─────────────────────────────────────
D2_LEARNING_RATE = 1e-5
D2_NUM_EPOCHS    = 5

# ─── DEPLOYMENT ───────────────────────────────────────────────────────────────
# Used ONLY at inference time. Never filter by this during training evaluation.
DEPLOY_CONFIDENCE_THRESHOLD = 0.6

# ─── PATHS — D1 ───────────────────────────────────────────────────────────────
# new_train.py reads all dataset.json files under preprocessed/ automatically.
# annotate_D1.py and preprocss_D1.py take --batch as a CLI argument.
# No batch names or merged file paths needed here.
D1_RAW_BASE          = "Invoice_dataset_D1/D1_raw"
D1_ANNOTATIONS_BASE  = "Invoice_dataset_D1/annotations"
D1_PREPROCESSED_BASE = "Invoice_dataset_D1/preprocessed"

# ─── PATHS — D2 ───────────────────────────────────────────────────────────────
D2_RAW_JSON    = "Into_the_wild_D2/dataset_wild_unmapped.json"
D2_MAPPED_JSON = "Into_the_wild_D2/dataset_mapped.json"

# ─── PATHS — SPLITS ───────────────────────────────────────────────────────────
SPLITS_DIR       = "splits"
TRAIN_SPLIT_JSON = os.path.join(SPLITS_DIR, "train_split.json")
VAL_SPLIT_JSON   = os.path.join(SPLITS_DIR, "val_split.json")
TEST_SPLIT_JSON  = os.path.join(SPLITS_DIR, "test_split.json")

# ─── PATHS — MODELS ───────────────────────────────────────────────────────────
BASE_MODEL       = "microsoft/layoutlm-base-uncased"
D1_MODEL_DIR     = "models/layoutlm_D1_final"
D2_MODEL_DIR     = "models/layoutlm_wild_final"

# ─── PATHS — RESULTS ──────────────────────────────────────────────────────────
RESULTS_DIR_D1   = "results/D1"
RESULTS_DIR_D2   = "results/D2"
SIMILARITY_DIR   = "similarity_results_1"