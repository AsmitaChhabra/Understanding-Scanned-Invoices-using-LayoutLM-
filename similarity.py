import json
import os
import numpy as np
import torch
from transformers import LayoutLMModel, LayoutLMTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────
RUNS = [
    {
        "name": "D1_invoices",
        "model_path": "models/layoutlm_D1_final",
        "dataset_json": "Invoice_dataset_D1/preprocessed/batch1_1/dataset.json",
        "output_file": "similarity_results_1/similarity_results_D1.json",
    },
    {
        "name": "D2_wild",
        "model_path": "models/layoutlm_wild_final",
        "dataset_json": "Into_the_wild_D2/dataset_mapped.json",
        "output_file": "similarity_results_1/similarity_results_D2.json",
    },
]

TOP_K                = 5
SIMILARITY_THRESHOLD = 0.80

os.makedirs("similarity_results_1", exist_ok=True)

# ─── DEVICE ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

# ─── JACCARD ────────────────────────────────────────────
def jaccard_similarity(words_a, words_b):
    set_a = set(w.lower() for w in words_a)
    set_b = set(w.lower() for w in words_b)
    if not (set_a | set_b):
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# ─── FEATURE EXTRACTION ─────────────────────────────────
def extract_features(sample, tokenizer, model):
    words = sample["words"]
    boxes = [[max(0, min(1000, c)) for c in box] for box in sample["bboxes"]]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids(batch_index=0)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    aligned_boxes = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_boxes.append(boxes[word_idx])

    bbox_tensor = torch.tensor([aligned_boxes], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            bbox=bbox_tensor
        )

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().cpu().numpy()

# ─── MAIN LOOP OVER BOTH DATASETS ───────────────────────
for run in RUNS:
    print(f"\n{'='*55}")
    print(f"🚀 Running similarity for: {run['name']}")
    print(f"{'='*55}")

    # Load model for this run
    print("📦 Loading model...")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(
        run["model_path"], local_files_only=True
    )
    model = LayoutLMModel.from_pretrained(
        run["model_path"], local_files_only=True
    )
    model.to(device)
    model.eval()

    # Load dataset
    with open(run["dataset_json"], "r") as f:
        dataset = json.load(f)

    print(f"📂 Loaded {len(dataset)} samples from {run['dataset_json']}")

    # Extract features
    all_features, all_words, all_file_names = [], [], []

    for i, sample in enumerate(tqdm(dataset, desc="Extracting features")):
        features = extract_features(sample, tokenizer, model)
        all_features.append(features)
        all_words.append(sample["words"])
        all_file_names.append(sample.get("id", f"invoice_{i}"))

    feature_matrix = np.vstack([f.squeeze(0) for f in all_features])

    # Cosine similarity
    print("📐 Computing cosine similarity...")
    cosine_matrix = cosine_similarity(feature_matrix)

    # Jaccard similarity
    print("📐 Computing Jaccard similarity...")
    n = len(all_words)
    jaccard_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Jaccard"):
        for j in range(i + 1, n):
            score = jaccard_similarity(all_words[i], all_words[j])
            jaccard_matrix[i][j] = score
            jaccard_matrix[j][i] = score
        jaccard_matrix[i][i] = 1.0

    # Build results
    results = []
    for i in range(n):
        top_k_cosine = sorted(
            [(j, float(cosine_matrix[i][j])) for j in range(n) if j != i],
            key=lambda x: x[1], reverse=True
        )[:TOP_K]

        top_k_jaccard = sorted(
            [(j, float(jaccard_matrix[i][j])) for j in range(n) if j != i],
            key=lambda x: x[1], reverse=True
        )[:TOP_K]

        results.append({
            "query_invoice": all_file_names[i],
            "top_similar_cosine": [
                {
                    "invoice": all_file_names[j],
                    "score": round(score, 4),
                    "similar": score >= SIMILARITY_THRESHOLD
                }
                for j, score in top_k_cosine
            ],
            "top_similar_jaccard": [
                {
                    "invoice": all_file_names[j],
                    "score": round(score, 4),
                    "similar": score >= SIMILARITY_THRESHOLD
                }
                for j, score in top_k_jaccard
            ]
        })

    # Save
    with open(run["output_file"], "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done — {run['name']}")
    print(f"   Results saved: {run['output_file']}")

    # Summary
    cosine_pairs  = sum(1 for r in results for m in r["top_similar_cosine"]  if m["similar"])
    jaccard_pairs = sum(1 for r in results for m in r["top_similar_jaccard"] if m["similar"])
    print(f"   Total invoices     : {n}")
    print(f"   Cosine similar pairs : {cosine_pairs}")
    print(f"   Jaccard similar pairs: {jaccard_pairs}")

    # Free memory before next run
    del model, tokenizer
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

print("\n🎉 Both datasets complete.")
