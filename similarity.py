import json
import os
import glob
import numpy as np
import torch
from transformers import LayoutLMModel, LayoutLMTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config_D2 import (
    MAX_SEQ_LEN,
    D1_PREPROCESSED_BASE,
    D2_MAPPED_JSON,
    D1_MODEL_DIR,
    D2_MODEL_DIR,
    SIMILARITY_DIR,
)

# =========================================================
# SIMILARITY CONFIG
# =========================================================

TOP_K = 5

# Per-dataset thresholds — cosine similarity behaves
# differently depending on how visually/textually similar
# the underlying documents are.
#
# D1 invoices share a near-identical template, so embeddings
# cluster tightly (observed mean cosine ~0.98). A 0.80
# threshold flags almost everything as "similar" — not useful.
# D1 needs a much stricter threshold to surface genuine
# duplicates rather than just "same template".
#
# D2 receipts are visually diverse (different vendors,
# layouts), so cosine scores spread out more naturally
# (observed mean cosine ~0.88). 0.80 is meaningful here.
COSINE_THRESHOLD = {
    "D1_invoices": 0.995,
    "D2_wild":     0.80,
}
JACCARD_THRESHOLD = {
    "D1_invoices": 0.80,
    "D2_wild":     0.80,
}

# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

# =========================================================
# JACCARD SIMILARITY
# Word-level overlap between two invoices.
# Useful for detecting near-duplicate documents.
# =========================================================

def jaccard_similarity(words_a, words_b):
    set_a = set(w.lower() for w in words_a)
    set_b = set(w.lower() for w in words_b)
    if not (set_a | set_b):
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# =========================================================
# FEATURE EXTRACTION
# Uses LayoutLM as a feature extractor — mean pools the
# last hidden state across all tokens to get a single
# embedding per invoice that captures both layout and text.
# token_type_ids included — required by LayoutLM.
# =========================================================

def extract_features(sample, tokenizer, model):
    words = sample["words"]
    boxes = [[max(0, min(1000, c)) for c in box] for box in sample["bboxes"]]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

    word_ids = encoding.word_ids(batch_index=0)

    aligned_boxes = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_boxes.append(boxes[word_idx])

    bbox_tensor = torch.tensor([aligned_boxes], dtype=torch.long).to(device)
    encoding    = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=      encoding["input_ids"],
            attention_mask= encoding["attention_mask"],
            token_type_ids= encoding["token_type_ids"],   # required by LayoutLM
            bbox=           bbox_tensor,
        )

    # Mean pool across all token positions → single invoice embedding
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().cpu().numpy()

# =========================================================
# LOAD D1 DATASET
# Auto-discovers all dataset.json files under preprocessed/
# Same logic as new_train.py — no hardcoded batch paths.
# =========================================================

def load_d1_dataset():
    dataset_files = sorted(glob.glob(
        os.path.join(D1_PREPROCESSED_BASE, "**", "preprocessed_dataset.json"),
        recursive=True,
    ))

    if not dataset_files:
        raise FileNotFoundError(
            f"No dataset.json files found under {D1_PREPROCESSED_BASE}"
        )

    data = []
    for path in dataset_files:
        with open(path, "r") as f:
            batch = json.load(f)
        print(f"   {path} — {len(batch)} samples")
        data += batch

    return data

# =========================================================
# GET SAMPLE ID
# D1 samples use file_name, D2 samples use id.
# Falls back to invoice_N index if neither exists.
# =========================================================

def get_sample_id(sample, index):
    return (
        sample.get("file_name")
        or sample.get("id")
        or f"invoice_{index}"
    )

# =========================================================
# RUN SIMILARITY FOR ONE DATASET
# =========================================================

def run_similarity(name, model_path, dataset, output_file):
    print(f"\n{'='*55}")
    print(f"🚀 Running similarity for: {name}")
    print(f"{'='*55}")

    cosine_threshold  = COSINE_THRESHOLD[name]
    jaccard_threshold = JACCARD_THRESHOLD[name]
    print(f"   Cosine threshold  : {cosine_threshold}")
    print(f"   Jaccard threshold : {jaccard_threshold}")

    # ── Load model ─────────────────────────────────────────
    print("📦 Loading model...")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(
        model_path, local_files_only=True
    )
    model = LayoutLMModel.from_pretrained(
        model_path, local_files_only=True
    )
    model.to(device)
    model.eval()

    print(f"📂 Loaded {len(dataset)} samples")

    # ── Extract features ───────────────────────────────────
    all_features  = []
    all_words     = []
    all_ids       = []

    for i, sample in enumerate(tqdm(dataset, desc="Extracting features")):
        features = extract_features(sample, tokenizer, model)
        all_features.append(features)
        all_words.append(sample["words"])
        all_ids.append(get_sample_id(sample, i))

    feature_matrix = np.vstack([f.squeeze(0) for f in all_features])

    # ── Cosine similarity ──────────────────────────────────
    print("📐 Computing cosine similarity...")
    cosine_matrix = cosine_similarity(feature_matrix)

    # ── Jaccard similarity ─────────────────────────────────
    print("📐 Computing Jaccard similarity...")
    n              = len(all_words)
    jaccard_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Jaccard"):
        for j in range(i + 1, n):
            score               = jaccard_similarity(all_words[i], all_words[j])
            jaccard_matrix[i][j] = score
            jaccard_matrix[j][i] = score
        jaccard_matrix[i][i] = 1.0

    # ── Build results ──────────────────────────────────────
    results          = []
    exact_duplicates = []   # cosine AND jaccard both above threshold

    for i in range(n):
        top_k_cosine = sorted(
            [(j, float(cosine_matrix[i][j])) for j in range(n) if j != i],
            key=lambda x: x[1], reverse=True,
        )[:TOP_K]

        top_k_jaccard = sorted(
            [(j, float(jaccard_matrix[i][j])) for j in range(n) if j != i],
            key=lambda x: x[1], reverse=True,
        )[:TOP_K]

        results.append({
            "query_invoice": all_ids[i],
            "top_similar_cosine": [
                {
                    "invoice": all_ids[j],
                    "score":   round(score, 4),
                    "similar": score >= cosine_threshold,
                }
                for j, score in top_k_cosine
            ],
            "top_similar_jaccard": [
                {
                    "invoice": all_ids[j],
                    "score":   round(score, 4),
                    "similar": score >= jaccard_threshold,
                }
                for j, score in top_k_jaccard
            ],
        })

        # Flag likely duplicates — agreement between both metrics
        # is a much stronger signal than either alone.
        for j, c_score in top_k_cosine:
            if c_score >= cosine_threshold and jaccard_matrix[i][j] >= jaccard_threshold:
                if i < j:   # avoid double-counting symmetric pairs
                    exact_duplicates.append({
                        "invoice_a":    all_ids[i],
                        "invoice_b":    all_ids[j],
                        "cosine":       round(c_score, 4),
                        "jaccard":      round(float(jaccard_matrix[i][j]), 4),
                    })

    # ── Save ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "threshold_used": {
                "cosine":  cosine_threshold,
                "jaccard": jaccard_threshold,
            },
            "likely_duplicates": exact_duplicates,
            "results": results,
        }, f, indent=2)

    # ── Summary ────────────────────────────────────────────
    cosine_pairs  = sum(1 for r in results for m in r["top_similar_cosine"]  if m["similar"])
    jaccard_pairs = sum(1 for r in results for m in r["top_similar_jaccard"] if m["similar"])

    print(f"\n✅ Done — {name}")
    print(f"   Results saved          : {output_file}")
    print(f"   Total invoices         : {n}")
    print(f"   Cosine similar pairs   : {cosine_pairs}")
    print(f"   Jaccard similar pairs  : {jaccard_pairs}")
    print(f"   Likely duplicates      : {len(exact_duplicates)}  (cosine + jaccard agree)")
    if exact_duplicates:
        print(f"   Top duplicate pairs:")
        for dup in sorted(exact_duplicates, key=lambda d: d["cosine"], reverse=True)[:5]:
            print(f"     {dup['invoice_a']} ↔ {dup['invoice_b']}  "
                  f"(cosine: {dup['cosine']}, jaccard: {dup['jaccard']})")

    # ── Free memory ────────────────────────────────────────
    del model, tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(SIMILARITY_DIR, exist_ok=True)

    # ── D1 ─────────────────────────────────────────────────
    print(f"\n📂 Loading D1 datasets from: {D1_PREPROCESSED_BASE}")
    d1_dataset = load_d1_dataset()
    print(f"   Total D1 samples: {len(d1_dataset)}")

    run_similarity(
        name        = "D1_invoices",
        model_path  = D1_MODEL_DIR,
        dataset     = d1_dataset,
        output_file = os.path.join(SIMILARITY_DIR, "similarity_results_D1.json"),
    )

    # ── D2 ─────────────────────────────────────────────────
    print(f"\n📂 Loading D2 dataset: {D2_MAPPED_JSON}")
    with open(D2_MAPPED_JSON, "r") as f:
        d2_dataset = json.load(f)
    print(f"   Total D2 samples: {len(d2_dataset)}")

    run_similarity(
        name        = "D2_wild",
        model_path  = D2_MODEL_DIR,
        dataset     = d2_dataset,
        output_file = os.path.join(SIMILARITY_DIR, "similarity_results_D2.json"),
    )

    print("\n🎉 Both datasets complete.")


if __name__ == "__main__":
    main()